#version 330

// Input vertex attributes (from vertex shader)
in vec2 fragTexCoord;
in vec4 fragColor;

// Input uniform values
uniform sampler2D texture0;
uniform vec4 colDiffuse;

// Output fragment color
out vec4 finalColor;

void main()
{
    // Calculate vector from center (0.5, 0.5)
    vec2 center = vec2(0.5, 0.5);
    vec2 p = fragTexCoord - center;

    // Calculate distance squared (faster than length)
    float r2 = dot(p, p);

    // Discard pixels outside the circle (radius 0.5 -> r^2 = 0.25)
    if (r2 > 0.25) discard;

    // Gaussian falloff
    // Standard Gaussian: e^(-x^2)
    // We adjust the coefficient to ensure it fades to ~0 at the edges (r=0.5, r2=0.25)
    float alpha = exp(-r2 * 20.0);

    // Apply texture color (if any), vertex color, and calculated alpha
    // We assume texture0 is white if not bound, so we just use fragColor
    vec4 texColor = texture(texture0, fragTexCoord);

    finalColor = texColor * fragColor * colDiffuse;
    finalColor.a *= alpha;
}
