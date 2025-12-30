#version 330

in vec2 fragTexCoord;
in vec4 fragColor;
uniform vec4 colDiffuse;
out vec4 color;

void main()
{
    vec2 center = vec2(0.5, 0.5);
    vec2 p = fragTexCoord - center;
    float r2 = dot(p, p);
    if (r2 > 0.25) discard;
    float alpha = exp(-r2 * 20.0);
    color = fragColor;
    color.rgb = pow(color.rgb, vec3(1.0/2.2));
    color.a *= alpha;
}
