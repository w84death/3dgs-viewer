#version 330

// Input vertex attributes (from vertex shader)
in vec2 fragTexCoord;
in vec4 fragColor;

// Input uniform values
uniform sampler2D texture0;
uniform vec4 colDiffuse;

// Custom uniforms for lighting
uniform vec3 lightDir; // Direction TO the light
uniform vec3 viewPos;  // Camera position (world space)

// Output fragment color
out vec4 finalColor;

// Narkowicz ACES Tone Mapping
vec3 ACESFilm(vec3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e), 0.0, 1.0);
}

void main()
{
    // 1. Calculate soft circular shape (Gaussian Splat geometry)
    vec2 center = vec2(0.5, 0.5);
    vec2 p = fragTexCoord - center;
    float r2 = dot(p, p);

    // Discard corners to make it round
    if (r2 > 0.25) discard;

    // 2. Reconstruct Normal (Spherical Impostor)
    // Map p from [-0.5, 0.5] to [-1.0, 1.0] for the XY components of the normal
    vec3 N;
    N.xy = p * 2.0;
    // Calculate Z based on unit sphere equation: x^2 + y^2 + z^2 = 1
    float mag = dot(N.xy, N.xy);
    if (mag > 1.0) discard;
    N.z = sqrt(1.0 - mag);
    N = normalize(N);

    // 3. Lighting Calculation (Blinn-Phong)

    // Ambient
    vec3 ambient = vec3(0.3, 0.3, 0.4);

    // Diffuse
    vec3 L = normalize(lightDir);
    float diff = max(dot(N, L), 0.0);
    // Boost diffuse intensity slightly for tone mapping
    vec3 diffuse = diff * vec3(1.5, 1.4, 1.2);

    // Specular
    // View vector approximation (Viewer is roughly along Z for these billboards)
    vec3 V = vec3(0.0, 0.0, 1.0);
    vec3 H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), 32.0);
    vec3 specular = vec3(1.0) * spec * 0.8;

    vec3 lighting = ambient + diffuse + specular;

    // 4. Base Color
    vec4 texColor = texture(texture0, fragTexCoord);
    vec3 albedo = texColor.rgb * fragColor.rgb * colDiffuse.rgb;

    // Apply lighting
    vec3 color = albedo * lighting;

    // 5. Tone Mapping (ACES)
    color = ACESFilm(color);

    // 6. Gamma Correction
    color = pow(color, vec3(1.0/2.2));

    // 7. Gaussian Alpha Falloff
    float alpha = exp(-r2 * 10.0);

    finalColor = vec4(color, fragColor.a * alpha);
}
