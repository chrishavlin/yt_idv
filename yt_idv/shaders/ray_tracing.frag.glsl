in vec4 v_model;
flat in vec3 dx;
flat in vec3 left_edge;
flat in vec3 right_edge;
flat in mat4 inverse_proj;
flat in mat4 inverse_mvm;
flat in mat4 inverse_pmvm;
flat in ivec3 texture_offset;
out vec4 output_color;

bool within_bb(vec3 pos)
{
    bvec3 left =  greaterThanEqual(pos, left_edge);
    bvec3 right = lessThanEqual(pos, right_edge);
    return all(left) && all(right);
}

vec3 get_offset_texture_position(sampler3D tex, vec3 tex_curr_pos)
{
    ivec3 texsize = textureSize(tex, 0); // lod (mipmap level) always 0?
    return (tex_curr_pos * texsize + texture_offset) / texsize;
}

bool sample_texture(vec3 tex_curr_pos, inout vec4 curr_color, float tdelta,
                    float t, vec3 dir);
vec4 cleanup_phase(in vec4 curr_color, in vec3 dir, in float t0, in float t1);

// This main() function will call a function called sample_texture at every
// step along the ray.  It must be of the form
//   void (vec3 tex_curr_pos, inout vec4 curr_color, float tdelta, float t,
//         vec3 direction);

void main()
{
    // Obtain screen coordinates
    // https://www.opengl.org/wiki/Compute_eye_space_from_window_space#From_gl_FragCoord
    vec3 ray_position = v_model.xyz;


    // Five samples
    vec3 step_size = dx/sample_factor;
    vec3 dir = -normalize(camera_pos.xyz - ray_position);
    dir = max(abs(dir), 0.0001) * sign(dir);
    vec4 curr_color = vec4(0.0);

    // We need to figure out where the ray intersects the box, if it intersects the box.
    // This will help solve the left/right edge issues.

    vec3 idir = 1.0/dir;
    vec3 tl = (left_edge - camera_pos)*idir;
    vec3 tr = (right_edge - camera_pos)*idir;
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
    vec2 UV = vec2(0.);
    vec4 prior_color = vec4(0.);
    if (p1_second_pass) {
        // sample the temporary framebuffer texture from the prior pass
        // viewport = vec4 of x0, y0, w, h

        // same full cube, data off centered
        // THIS WORKS. BUT WHY DO I NEED TO REVERSE xy. THAT IS WEIRD.
        // might be in the texture definition?
        UV.xy = gl_FragCoord.yx / viewport.zw;
        // uncomment hte following to short-circuit and return just the texture
        // value, ends up reproducing the original image
        // output_color = texture(fb_temp_tex, UV);
        // return;
        // the color of the pixel containing this fragment (from prior pass):
        prior_color = texture(fb_temp_tex, UV);
    }
    // Some more discussion of this here:
    //  http://prideout.net/blog/?p=64

    vec3 p0 = camera_pos.xyz + dir * t0;
    vec3 p1 = camera_pos.xyz + dir * t1;

    vec3 dxidir = abs(idir)  * step_size;

    temp_t = min(dxidir.xx, dxidir.yz);

    float tdelta = min(temp_t.x, temp_t.y);
    float t = t0;

    vec3 range = (right_edge + dx/2.0) - (left_edge - dx/2.0);
    vec3 nzones = range / dx;
    vec3 ndx = 1.0/nzones;

    vec3 tex_curr_pos = vec3(0.0);

    bool sampled;
    bool ever_sampled = false;

    vec4 v_clip_coord;
    float f_ndc_depth;
    float depth = 1.0;

    ray_position = p0;

    bool still_looking_for_max = true;
    bool found_max = false;

    while(t <= t1) {
        tex_curr_pos = (ray_position - left_edge) / range;  // Scale from 0 .. 1
        // But, we actually need it to be 0 + normalized dx/2 to 1 - normalized dx/2
        tex_curr_pos = (tex_curr_pos * (1.0 - ndx)) + ndx/2.0;

        sampled = sample_texture(tex_curr_pos, curr_color, tdelta, t, dir);

        if (sampled) {
            ever_sampled = true;
            if (p1_second_pass) {
                if (still_looking_for_max) {
                    if (prior_color.r > 0) {
                    float dcolor = abs(prior_color.r - curr_color.r);
                    float deps = 0.00001;
                    // floating point comparison issues? dcolor == 0 fails...
                        if (dcolor <= deps) {
                            // only compare r channel because the data value is stored
                            // in the r channel during program1 execution
                            v_clip_coord = projection * modelview * vec4(ray_position, 1.0);
                            f_ndc_depth = v_clip_coord.z / v_clip_coord.w; // from -1 to 1 now
                            depth = (1.0 - 0.0) * 0.5 * f_ndc_depth + (1.0 + 0.0) * 0.5;
                            still_looking_for_max = false;
                            found_max = true;
                            // should be safe to terminate the loop at this point, but
                            // going to let it keep running for now...
                        }
                    }
                }
            }
        }

        t += tdelta;
        ray_position += tdelta * dir;

    }

    if (p1_second_pass){
        // just pass along the color
        output_color = cleanup_phase(prior_color, dir, t0, t1);
    } else {
        output_color = cleanup_phase(curr_color, dir, t0, t1);
        // the output color for this fragment
        // final pixel will be the blend of all fragments at this pixel
    }

    if (ever_sampled) {
        if (p1_second_pass) {
            if (found_max){
                    // only set the fragment depth on the second pass when we know
                    // it is the depth of the max value
                    gl_FragDepth = depth;
                }
            }
    }
}
