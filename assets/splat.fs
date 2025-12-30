#version 330

in vec2 fragTexCoord;
in vec4 fragColor;


// Narkowicz ACES Tone Mapping
vec3 ACESFilm(vec3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e), 0.0, 1.0);
}
out vec4 color;

void main()
{
    // 1. Calculate soft circular shape (Gaussian Splat geometry)
    vec2 center = vec2(0.5, 0.5);
    vec2 p = fragTexCoord - center;
    float r2 = dot(p, p);

    if (r2 > 0.25) discard;
    color.rgb = ACESFilm(color.rgb);
    color.rgb = pow(color.rgb, vec3(1.0/2.2));

    float alpha = exp(-r2 * 20.0);
    color = fragColor;
    color.rgb = pow(color.rgb, vec3(1.0/2.2));
    color.a *= alpha;
}
