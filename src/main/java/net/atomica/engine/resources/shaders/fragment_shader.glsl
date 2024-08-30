#version 330 core

in vec3 FragPos;
in vec2 TexCoords;

out vec4 FragColor;

uniform sampler2D texture_diffuse1;

uniform float fogDensity = 0.045;

void main() {
    vec3 textureColor = texture(texture_diffuse1, TexCoords).rgb;

    FragColor = vec4(textureColor, 0.0);
}

//#version 330 core
//
//in vec3 FragPos;
//in vec3 Normal;
//in vec2 TexCoords;
//
//out vec4 FragColor;
//
//uniform sampler2D texture_diffuse1;
//uniform vec3 lightPos;
//uniform vec3 viewPos;
//uniform vec3 lightColor;
//
//uniform vec3 skyColor = vec3(0.53, 0.81, 0.92);  // Light blue
//uniform float fogDensity = 0.045;
//
//void main() {
//    float ambientStrength = 0.1;
//    vec3 ambient = ambientStrength * lightColor;
//
//    vec3 norm = normalize(Normal);
//    vec3 lightDir = normalize(lightPos - FragPos);
//    float diff = max(dot(norm, lightDir), 0.0);
//    vec3 diffuse = diff * lightColor;
//
//    vec3 viewDir = normalize(viewPos - FragPos);
//    vec3 reflectDir = reflect(-lightDir, norm);
//    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
//    vec3 specular = 0.5 * spec * lightColor;
//
//    vec3 result = (ambient + diffuse + specular) * texture(texture_diffuse1, TexCoords).rgb;
//
//    float distance = length(viewPos - FragPos);
//    float fogFactor = exp(-fogDensity * distance);
//    fogFactor = clamp(fogFactor, 0.0, 1.0);
//
//    vec3 finalColor = mix(skyColor, result, fogFactor);
//
//    FragColor = vec4(finalColor, 1.0);
//}