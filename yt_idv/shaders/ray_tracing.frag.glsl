in vec4 v_model;
flat in vec3 dx;
flat in vec3 left_edge;
flat in vec3 right_edge;
flat in mat4 inverse_proj;
flat in mat4 inverse_mvm;
flat in mat4 inverse_pmvm;
flat in ivec3 texture_offset;

#ifdef SPHERICAL_GEOM
flat in vec3 left_edge_cart;
flat in vec3 right_edge_cart;
flat in vec4 phi_plane_le;
flat in vec4 phi_plane_re;
#endif

out vec4 output_color;

bool within_bb(vec3 pos)
{
    bvec3 left =  greaterThanEqual(pos, left_edge);
    bvec3 right = lessThanEqual(pos, right_edge);
    return all(left) && all(right);
}

#ifdef SPHERICAL_GEOM

vec3 scale_cart(vec3 v) {
    vec3 vout;
    vout[0] = v[0] * cart_bbox_max_width + cart_bbox_le[0];
    vout[1] = v[1] * cart_bbox_max_width + cart_bbox_le[1];
    vout[2] = v[2] * cart_bbox_max_width + cart_bbox_le[2];
    return vout;
}

vec3 cart_to_sphere_vec3(vec3 v) {
    // transform a single point in cartesian coords to spherical
    vec3 vout;
    float phi;
    vec3 vsc = v;

    // in yt, phi is the polar angle from (0, 2pi), theta is the azimuthal
    // angle (0, pi). the id_ values below are uniforms that depend on the
    // yt dataset coordinate ordering, cart_bbox_* variables are also uniforms
    vout[id_r] = vsc[0] * vsc[0] + vsc[1] * vsc[1] + vsc[2] * vsc[2];
    vout[id_r] = sqrt(vout[id_r]);
    vout[id_theta] = acos(vsc[2] / vout[id_r]);
    phi = atan(vsc[1], vsc[0]); // atan will retrun in -pi/2 to pi/2
    vout[id_phi] = phi + 2.0 * PI * float(phi < 0.0);
    return vout;
}

float get_ray_plane_intersection(vec3 p_normal, float p_constant, vec3 ray_origin, vec3 ray_dir)
{
    // returns a single float. if the ray is parallel to the plane, will return a null value
    // of -99.
    float n_dot_u = dot(p_normal, ray_dir);
    
    if (n_dot_u == 0.0){
        return -99.;
    } else {
        float n_dot_ro = dot(p_normal, ray_origin);
        return (p_constant - n_dot_ro) / n_dot_u ; 
    }

    // float n_dot_is_zero = float(n_dot_u == 0.0);    
    // return ((p_constant - n_dot_ro) / n_dot_u) * (1.0 - n_dot_is_zero) - 99. * n_dot_is_zero;
}

vec2 quadratic_eval(float b, float a_2, float c){
    // evalulate the quadratic equation
    //    (-b +/- sqrt(b^2 - 4*a*c)) / (2 * a)
    // or, in terms of the inputs to this function:
    //    (-b +/- sqrt(b^2 - 2*a_2*c)) / a_2
    //
    // Always returns vector of 2 values, but if the determinate (b^2 - 4ac) is:
    // negative: both values will be null placeholders, -99.
    // exactly 0: second value will be null placeholder, -99.
    // positive: both values are real solutions

    float det = b*b - 2.0 * a_2 * c; // determinate

    // handle determinate cases via bool multiplications
    // det == 0 : 1 intersection
    // det > 0 : 2 real intersections.
    //     Note that real intersections may be with the shadow cone! but those intersections will
    //     end up at a t outside the cartesian bounding box
    // det < 0 : no intersections
    // float det_is_nonneg = 1.0 * float(det >= 0.);
    // float det_is_zero = 1.0 * float(det == 0.);
    // float null_val = -99. *  (1. - det_is_nonneg); // -99. if negative, 1.0 if positive or 0.
    // det = det * det_is_nonneg;
    // do the calculation
    if (det <= 0.){ 
        return vec2(-99., -99.);
    } else if (det == 0){ 
        return vec2(-b / a_2, -99);
    } else { 
        return vec2((-b - sqrt(det)) / a_2, (-b + sqrt(det)) / a_2);
    }
    // vec2 return_vec;
    // return_vec = vec2((-b - sqrt(det)) / a_2 * det_is_nonneg + null_val, (-b + sqrt(det)) / a_2 * det_is_nonneg + null_val);
    // // override the second return if det is 0.
    // return_vec[1] = return_vec[1] * (1. - det_is_zero)  - 99. * det_is_zero;

    // return return_vec;
}

vec2 get_ray_sphere_intersection(float r, vec3 ray_origin, vec3 ray_dir)
{
    // assumes sphere is centered at (x, y, z) = (0, 0, 0)
    float a_2 = 2.0 * dot(ray_dir, ray_dir);
    float b = 2.0 * dot(ray_dir, ray_origin);
    float c = dot(ray_origin, ray_origin) - r * r;

    return quadratic_eval(b, a_2, c);
}

vec2 get_ray_cone_intersection(float theta, vec3 ray_origin, vec3 ray_dir)
{
    // returns a vec2 containing 0, 1 or 2 intersections. Null values are indicated
    // by negative placeholder numbers.
    // theta : the fixed theta value defining the z-aligned cone 
    // ray_origin : ray origin 
    // ray_dir : ray direction 
    //
    // note: it is possible to have real solutions that intersect the shadow cone
    // and not the actual cone. but those values will end up outside the t range
    // given by intersection with the bounding cartesian box, so we do not need
    // to handle them explicitly here.
    // 
    // See the following for some more background 
    // https://github.com/chrishavlin/miscellaneous_python/blob/main/notebooks/spherical_volumes.ipynb    

    float costheta;
    vec3 vhat; // cone axis 

    // if theta is past PI/2, the cone will point in negative z and the
    // half angle should be measured from the -z axis, not +z.
    // also note that theta = PI/2.0 is well defined. determinate = 0 in that case and
    // the cone becomes a plane in x-y.
    // float theta_pi2 = 1.0 - 2.0 * float(theta > PI/2.0); // 1.0 if theta <= PI/2, -1 else
    float theta_pi2;
    if (theta <= PI/ 2.0){ 
        theta_pi2 = 1.0;
        vhat = vec3(0., 0., 1.0); 
    } else {
        theta_pi2 = -1.0;
        vhat = vec3(0., 0., -1.0);
    }
    // vhat = vec3(0.0, 0.0, theta_pi2); // (0,0,1) or (0,0,-1)
    costheta = theta_pi2 * cos(theta); // trig identity cos(PI - theta) = - cos(theta)
    float cos2t = pow(costheta, 2);
    float dir_dot_vhat = dot(ray_dir, vhat);
    float dir_dot_dir = dot(ray_dir, ray_dir); // equivalent to np.linalg.norm(u)**2
    float ro_dot_vhat = dot(ray_origin, vhat);
    float ro_dot_dir = dot(ray_origin, ray_dir);
    float ro_dot_ro = dot(ray_origin, ray_origin);

    float a_2 = 2.0*(pow(dir_dot_vhat, 2) - dir_dot_dir * cos2t);
    float b = 2.0 * (ro_dot_vhat * dir_dot_vhat - ro_dot_dir*cos2t);
    float c = pow(ro_dot_vhat, 2) - ro_dot_ro*cos2t;

    return quadratic_eval(b, a_2, c);
}

int store_temp_intx(int n_extra, vec4 t_extra, float t_temp, float t0, float t1) {
    // store a possible intersection with a spherical volume element primitive
    // geometry if it falls within the initial t range from the cartesian bounding
    // box intersection
    if (t_temp > t0 && t_temp < t1){
        t_extra[n_extra] = t_temp;
        return 1;
    }
    return 0;
}


float max_of_vec3(vec4 in_vec){
    float temp_val1 = max(in_vec[0], in_vec[1]);    
    return max(temp_val1, in_vec[2]);
}


float min_of_vec3(vec4 in_vec){
    float temp_val1 = min(in_vec[0], in_vec[1]);    
    return min(temp_val1, in_vec[2]);
}


float max_of_vec4(vec4 in_vec){
    float temp_val1 = max(in_vec[0], in_vec[1]);
    float temp_val2 = max(in_vec[2], in_vec[3]);
    return max(temp_val1, temp_val2);
}

float min_of_vec4(vec4 in_vec){
    float temp_val1 = min(in_vec[0], in_vec[1]);
    float temp_val2 = min(in_vec[2], in_vec[3]);
    return min(temp_val1, temp_val2);
}

#endif

vec3 get_offset_texture_position(sampler3D tex, vec3 tex_curr_pos)
{
    ivec3 texsize = textureSize(tex, 0); // lod (mipmap level) always 0?
    return (tex_curr_pos * texsize + texture_offset) / texsize;
}

bool sample_texture(vec3 tex_curr_pos, inout vec4 curr_color, float tdelta,
                    float t, vec3 dir);
vec4 cleanup_phase(in vec4 curr_color, in vec3 dir, in float t0, in float t1);

// This main() function will call a function called sample_texture at every
// step along the ray.  sample_texture must be of the form
//   void (vec3 tex_curr_pos, inout vec4 curr_color, float tdelta, float t,
//         vec3 direction);
void main()
{

    // Obtain screen coordinates
    // https://www.opengl.org/wiki/Compute_eye_space_from_window_space#From_gl_FragCoord
    vec3 ray_position = v_model.xyz;
    vec3 camera_pos_data = camera_pos.xyz;
    #ifdef SPHERICAL_GEOM
    // this ensures any bounding box scaling is accounted for, which is required
    // for the ray intersections with the spherical volume element
    ray_position = scale_cart(ray_position);
    camera_pos_data = scale_cart(camera_pos_data);
    #endif
    vec3 ray_position_native;

    output_color = vec4(0.);

    // Five samples
    vec3 dir = -normalize(camera_pos_data - ray_position);
    dir = max(abs(dir), 0.0001) * sign(dir);
    vec4 curr_color = vec4(0.0);

    // We need to figure out where the ray intersects the box, if it intersects the box.
    // This will help solve the left/right edge issues.

    vec3 idir = 1.0/dir;
    vec3 tl, tr, dx_cart;
    #ifdef SPHERICAL_GEOM
    tl = (scale_cart(left_edge_cart) - camera_pos_data)*idir;
    tr = (scale_cart(right_edge_cart) - camera_pos_data)*idir;
    dx_cart = scale_cart(right_edge_cart) - scale_cart(left_edge_cart);
    #else
    tl = (left_edge - camera_pos)*idir;
    tr = (right_edge - camera_pos)*idir;
    dx_cart = dx;
    #endif
    vec3 step_size = dx_cart / sample_factor;

    vec3 tmin, tmax;
    bvec3 temp_x, temp_y;
    // These 't' prefixes actually mean 'parameter', as we use in grid_traversal.pyx.

    tmax = vec3(lessThan(dir, vec3(0.0)))*tl+vec3(greaterThanEqual(dir, vec3(0.0)))*tr;
    tmin = vec3(greaterThanEqual(dir, vec3(0.0)))*tl+vec3(lessThan(dir, vec3(0.0)))*tr;
    vec2 temp_t = max(tmin.xx, tmin.yz);
    float t0 = max(temp_t.x, temp_t.y);

    // smallest tmax
    temp_t = min(tmax.xx, tmax.yz);
    float t1 = min(temp_t.x, temp_t.y);
    t0 = max(t0, 0.0);
    if (t1 <= t0) discard;

    vec3 dxidir = abs(idir)  * step_size;
    temp_t = min(dxidir.xx, dxidir.yz);
    float tdelta = min(temp_t.x, temp_t.y);

    vec4 t_control_points = vec4(-99.0);
    #ifdef SPHERICAL_GEOM
    // here, we check the intersections with the primitive geometries describing the
    // surfaces of the spherical volume element. The intersections are only saved if
    // they fall within the ray parameter range given by the initial slab test
    // there are 0, 1, 2 or 4 intersections possible. 4 intersections is annoying.
    vec2 t_temp2;
    int n_extra = 0;
    float t2 = -99.0;
    float t3 = -99.0;

    // outer sphere
    t_temp2 = get_ray_sphere_intersection(right_edge[id_r], camera_pos_data, dir);
    if (t_temp2[0] >= t0 && t_temp2[0] <= t1){
        t_control_points[n_extra] = t_temp2[0];
        n_extra = n_extra + 1;
    }
    if (t_temp2[1] >= t0 && t_temp2[1] <= t1){
        t_control_points[n_extra] = t_temp2[1];
        n_extra = n_extra + 1;
    }
    // inner sphere
    t_temp2 = get_ray_sphere_intersection(left_edge[id_r], camera_pos_data, dir);
    if (t_temp2[0] >= t0 && t_temp2[0] <= t1){
        t_control_points[n_extra] = t_temp2[0];
        n_extra = n_extra + 1;
    }
    if (t_temp2[1] >= t0 && t_temp2[1] <= t1){
        t_control_points[n_extra] = t_temp2[1];
        n_extra = n_extra + 1;
    }    

    // the phi-normal planes
    t_temp2[0] = get_ray_plane_intersection(vec3(phi_plane_le), phi_plane_le[3], camera_pos_data, dir);
    if (t_temp2[0] >= t0 && t_temp2[0] <= t1){
        t_control_points[n_extra] = t_temp2[0];
        n_extra = n_extra + 1;
    }
    t_temp2[0] = get_ray_plane_intersection(vec3(phi_plane_re), phi_plane_re[3], camera_pos_data, dir);
    if (t_temp2[0] >= t0 && t_temp2[0] <= t1){
        t_control_points[n_extra] = t_temp2[0];
        n_extra = n_extra + 1;
    }

    // the fixed-theta cones
    t_temp2 = get_ray_cone_intersection(right_edge[id_theta], camera_pos_data, dir);
    if (t_temp2[0] >= t0 && t_temp2[0] <= t1){
        t_control_points[n_extra] = t_temp2[0];
        n_extra = n_extra + 1;
    }
    if (t_temp2[1] >= t0 && t_temp2[1] <= t1){
        t_control_points[n_extra] = t_temp2[1];
        n_extra = n_extra + 1;
    }   

    t_temp2 = get_ray_cone_intersection(left_edge[id_theta], camera_pos_data, dir);
    if (t_temp2[0] >= t0 && t_temp2[0] <= t1){
        t_control_points[n_extra] = t_temp2[0];
        n_extra = n_extra + 1;
    }
    if (t_temp2[1] >= t0 && t_temp2[1] <= t1){
        t_control_points[n_extra] = t_temp2[1];
        n_extra = n_extra + 1;
    }


    // float full_min;
    // float full_max;

    if (n_extra == 2){
        t0 = min(t_control_points[0], t_control_points[1])+.0001;
        t1 = max(t_control_points[0], t_control_points[1])-.0001;
    } else if (n_extra == 4){ 
        t1 = max_of_vec4(t_control_points)-.0001;
        t0 = min_of_vec4(t_control_points)+.0001;
    } else if (n_extra == 3){
        t1 = max_of_vec3(t_control_points)-.0001;
        t0 = min_of_vec3(t_control_points)+.0001;
    } else if (n_extra == 1){
        // should sample once at t0.
        t0 = t_control_points[0]+.0001;
        t1 = t0 + tdelta * 0.01; 
    } else if (n_extra == 0){
        discard;
    } else { 
        discard;
    }

    #endif

    // t_control_points[0] = t0;
    // t_control_points[1] = t1;

    // #endif
    // if (n_extra > 1)
    // {
    //     output_color = vec4(.2, 0., 1., 1);
    //     return;
    // }
    // output_color = vec4(0.);
    // return;

    // setup texture coordinates (always in native coordinates)
    vec3 range = (right_edge + dx/2.0) - (left_edge - dx/2.0);
    vec3 nzones = range / dx;
    vec3 ndx = 1.0/nzones;

    vec3 tex_curr_pos = vec3(0.0);

    // initialize ray tracing loop variables

    vec3 p0;  // cartesian position at t = t0
    vec3 p1;  // cartesian position at t = t1
    float t;  // current value of ray parameter t

    bool sampled;
    bool ever_sampled = false;
    bool within_el = true;

    vec4 v_clip_coord;
    float f_ndc_depth;
    float depth = 1.0;

    // Some more discussion of this here:
    //  http://prideout.net/blog/?p=64
    // for (int ipart=0;ipart<2;++ipart)
    // {
    //     t0 = t_control_points[ipart + 2 * ipart];  // index 0 or 2
    //     t1 = t_control_points[ipart + 1 + 2 * ipart]; // index 1 or 3
    //     t0 = max(t0, 0.0); // if t0 = -99., will have t > t1 and loop wont run below.
    p0 = camera_pos_data + dir * t0;
    p1 = camera_pos_data + dir * t1;
    t = t0;

    ray_position = p0;

    while(t <= t1) {

        // texture position
        #ifdef SPHERICAL_GEOM
        ray_position_native = cart_to_sphere_vec3(ray_position);
        within_el = within_bb(ray_position_native);
        #else
        ray_position_native = ray_position;
        #endif

        if (within_el) {
            tex_curr_pos = (ray_position_native - left_edge) / range;  // Scale from 0 .. 1
            // But, we actually need it to be 0 + normalized dx/2 to 1 - normalized dx/2
            tex_curr_pos = (tex_curr_pos * (1.0 - ndx)) + ndx/2.0;
            sampled = sample_texture(tex_curr_pos, curr_color, tdelta, t, dir);
        }

        if (sampled) {
            ever_sampled = true;
            v_clip_coord = projection * modelview * vec4(ray_position, 1.0);
            f_ndc_depth = v_clip_coord.z / v_clip_coord.w;
            depth = min(depth, (1.0 - 0.0) * 0.5 * f_ndc_depth + (1.0 + 0.0) * 0.5);
        }

        t += tdelta;
        ray_position += tdelta * dir;

    }

    // }


    output_color = cleanup_phase(curr_color, dir, t0, t1);

    if (ever_sampled) {
        gl_FragDepth = depth;
    }
}
