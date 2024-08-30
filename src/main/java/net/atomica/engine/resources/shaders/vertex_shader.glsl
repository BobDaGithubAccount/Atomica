//#version 330 core
//
//layout(location = 0) in vec3 aPos;
//layout(location = 1) in vec3 aNormal;
//layout(location = 2) in vec2 aTexCoords;
//
//uniform mat4 model;
//uniform mat4 view;
//uniform mat4 projection;
//
//out vec3 FragPos;
//out vec3 Normal;
//out vec2 TexCoords;
//
//void main() {
//    // Transform the vertex position to world space
//    FragPos = vec3(model * vec4(aPos, 1.0));
//    // Transform the normal to world space and normalize it
//    Normal = mat3(transpose(inverse(model))) * aNormal;
//    // Pass the texture coordinates to the fragment shader
//    TexCoords = aTexCoords;
//    // Transform the vertex position to clip space
//    gl_Position = projection * view * vec4(FragPos, 1.0);
//}

#version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 2) in vec2 aTexCoords;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 FragPos;
out vec2 TexCoords;

void main() {
    FragPos = vec3(model * vec4(aPos, 1.0));
    TexCoords = aTexCoords;
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
