layout ( points ) in;
layout ( triangle_strip, max_vertices = 14 ) out;

// note: all in/out variables below are always in native coordinates (e.g.,
// spherical or cartesian) except when noted.
flat in vec3 vdx[];
flat in vec3 vleft_edge[];
flat in vec3 vright_edge[];
flat in vec3 vleft_edge_cart[];
flat in vec3 vright_edge_cart[];
flat in vec4 vphi_plane_le[];
flat in vec4 vphi_plane_re[];

flat out vec3 dx;
flat out vec3 left_edge;
flat out vec3 right_edge;
flat out vec3 left_edge_cart;
flat out vec3 right_edge_cart;
flat out mat4 inverse_proj; // always cartesian
flat out mat4 inverse_mvm; // always cartesian
flat out mat4 inverse_pmvm; // always cartesian
out vec4 v_model;

flat in mat4 vinverse_proj[];  // always cartesian
flat in mat4 vinverse_mvm[]; // always cartesian
flat in mat4 vinverse_pmvm[]; // always cartesian
flat in vec4 vv_model[];

flat out ivec3 texture_offset;

flat out vec4 phi_plane_le;
flat out vec4 phi_plane_re;

// https://stackoverflow.com/questions/28375338/cube-using-single-gl-triangle-strip
// suggests that the triangle strip we want for the cube is

uniform vec3 arrangement[8] = vec3[](
    vec3(0, 0, 0),
    vec3(1, 0, 0),
    vec3(0, 1, 0),
    vec3(1, 1, 0),
    vec3(0, 0, 1),
    vec3(1, 0, 1),
    vec3(0, 1, 1),
    vec3(1, 1, 1)
);

uniform int aindex[14] = int[](6, 7, 4, 5, 1, 7, 3, 6, 2, 4, 0, 1, 2, 3);


void main() {

    vec4 center = gl_in[0].gl_Position;  // always cartesian

    vec3 width = vec3(0.);
    vec3 le = vec3(0.0);
    if (is_spherical) {
        width = vright_edge_cart[0] - vleft_edge_cart[0];
        le = vleft_edge_cart[0];
    } else {
        width = vright_edge[0] - vleft_edge[0];
        le = vleft_edge[0];
    }


    vec4 newPos;
    vec3 current_pos;

    for (int i = 0; i < 14; i++) {
        // walks through each vertex of the triangle strip, emit it. need to
        // emit gl_Position in cartesian, but pass native coords out in v_model

        // primitive node positions are always in cartesian
        current_pos = vec3(le + width * arrangement[aindex[i]]);
        newPos = vec4(current_pos, 1.0);
        gl_Position = projection * modelview * newPos;
        v_model = vec4(current_pos, 1.0);

        // these are also in cartesian
        inverse_pmvm = vinverse_pmvm[0];
        inverse_proj = vinverse_proj[0];
        inverse_mvm = vinverse_mvm[0];

        // these will be in native geometry
        left_edge = vleft_edge[0];
        right_edge = vright_edge[0];
        dx = vdx[0];

        // additional vertex attributes related to noncartesian geometries
        phi_plane_le = vphi_plane_le[0];
        phi_plane_re = vphi_plane_re[0];
        left_edge_cart = vleft_edge_cart[0];
        right_edge_cart = vright_edge_cart[0];

        texture_offset = ivec3(0);
        EmitVertex();
    }

    // why no endprimitive?
}
