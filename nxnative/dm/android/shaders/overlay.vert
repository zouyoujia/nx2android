#version 450
// Android port: the on-screen gamepad, one textured quad per draw blended over the frame: the whole static layer, or a sprite (a stick's knob where the
// finger is, the hidden gamepad's toggle button) from the texture's sprite strip. pos: the quad in clip space (x0 y0 x1 y1), uv: its texture rectangle
layout(push_constant) uniform Q { vec4 pos; vec4 uv; float alpha; uint turn; } q;
layout(location = 0) out vec2 uv;
void main() {
    const int corner[6] = int[6](0, 1, 2, 2, 1, 3);
    vec2 c = vec2(corner[gl_VertexIndex] & 1, corner[gl_VertexIndex] >> 1);
    uv = mix(q.uv.xy, q.uv.zw, c); vec2 p = mix(q.pos.xy, q.pos.zw, c);
    if (q.turn == 1u) p = vec2(-p.y, p.x);
    else if (q.turn == 3u) p = vec2(p.y, -p.x);
    gl_Position = vec4(p, 0.0, 1.0);
}
