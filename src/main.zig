const Engine = @import("engine/core.zig").Engine;
const rl = @import("raylib");
const std = @import("std");
const WIDTH = 1280;
const HEIGHT = 800;

const CamState = struct {
    distance: f32,
    theta: f32,
    phi: f32,
    initial_theta: f32,
    initial_phi: f32,
    dragging: bool = false,
    mouse_start: rl.Vector2,
    theta_start: f32,
    phi_start: f32,
};

const Splat = struct {
    pos: [3]f32,
    r: u8,
    g: u8,
    b: u8,
    a: u8,
};

const PlyLoadResult = struct {
    ply_data: []u8,
    splats: []Splat,
    vertex_count: usize,
};

fn loadPly(allocator: std.mem.Allocator, path: []const u8) !PlyLoadResult {
    const ply_data = try std.fs.cwd().readFileAlloc(allocator, path, std.math.maxInt(usize));

    // Find header end
    const header_end = std.mem.indexOf(u8, ply_data, "end_header") orelse return error.InvalidPly;
    const header = ply_data[0..header_end];
    var data_start = header_end + "end_header".len;
    while (data_start < ply_data.len and (ply_data[data_start] == '\r' or ply_data[data_start] == '\n')) data_start += 1;

    // Parse vertex count and properties from header
    var vertex_count: usize = 0;
    var properties: std.ArrayListUnmanaged([]const u8) = .{};
    defer properties.deinit(allocator);
    var in_vertex = false;
    var lines_iter = std.mem.splitScalar(u8, header, '\n');
    while (lines_iter.next()) |raw_line| {
        const line = std.mem.trim(u8, raw_line, "\r");
        if (std.mem.startsWith(u8, line, "element vertex ")) {
            var parts = std.mem.splitAny(u8, line, " ");
            _ = parts.next(); // "element"
            _ = parts.next(); // "vertex"
            if (parts.next()) |count_str| {
                vertex_count = try std.fmt.parseInt(usize, count_str, 10);
            }
            in_vertex = true;
        } else if (std.mem.startsWith(u8, line, "element ")) {
            in_vertex = false;
        } else if (in_vertex and std.mem.startsWith(u8, line, "property float ")) {
            var parts = std.mem.splitAny(u8, line, " ");
            _ = parts.next(); // "property"
            _ = parts.next(); // "float"
            if (parts.next()) |name| {
                try properties.append(allocator, name);
            }
        }
    }
    if (vertex_count == 0) return error.InvalidPly;

    const data = ply_data[data_start..];

    const splats = try allocator.alloc(Splat, vertex_count);
    std.debug.print("Loading binary PLY file with {} vertices...\n", .{vertex_count});
    // Binary parsing
    const stride = properties.items.len;
    const vertex_data_size = vertex_count * stride * 4;
    if (vertex_data_size > data.len) return error.InvalidData;
    const f32_slice = std.mem.bytesAsSlice(f32, data[0..vertex_data_size]);
    for (0..vertex_count) |ii| {
        const off = ii * stride;
        var pos: [3]f32 = undefined;
        var color_f: [3]f32 = undefined;
        var opacity: f32 = 0;
        for (properties.items, 0..) |name, idx| {
            const val = f32_slice[off + idx];
            if (std.mem.eql(u8, name, "x")) pos[0] = val else if (std.mem.eql(u8, name, "y")) pos[1] = val else if (std.mem.eql(u8, name, "z")) pos[2] = val else if (std.mem.eql(u8, name, "f_dc_0")) color_f[0] = val else if (std.mem.eql(u8, name, "f_dc_1")) color_f[1] = val else if (std.mem.eql(u8, name, "f_dc_2")) color_f[2] = val else if (std.mem.eql(u8, name, "opacity")) opacity = val;
        }
        const r_val = std.math.clamp((0.5 + color_f[0]) * 255.0, 0.0, 255.0);
        const g_val = std.math.clamp((0.5 + color_f[1]) * 255.0, 0.0, 255.0);
        const b_val = std.math.clamp((0.5 + color_f[2]) * 255.0, 0.0, 255.0);
        const a_val = std.math.clamp((1.0 / (1.0 + std.math.exp(-opacity))) * 255.0, 0.0, 255.0);
        const r = @as(u8, @intFromFloat(r_val));
        const g = @as(u8, @intFromFloat(g_val));
        const b = @as(u8, @intFromFloat(b_val));
        const a = @as(u8, @intFromFloat(a_val));
        splats[ii] = Splat{
            .pos = pos,
            .r = r,
            .g = g,
            .b = b,
            .a = a,
        };
        if (ii % 100000 == 0 or ii == vertex_count - 1) std.debug.print("Loaded {}/{} vertices\n", .{ ii + 1, vertex_count });
    }
    std.debug.print("Finished loading {} vertices from binary PLY\n", .{vertex_count});

    return PlyLoadResult{
        .ply_data = ply_data,
        .splats = splats,
        .vertex_count = vertex_count,
    };
}

const CameraInitResult = struct {
    camera: rl.Camera3D,
    cam_state: CamState,
};

fn initCamera(center: [3]f32) CameraInitResult {
    const distance = 1.0;
    const theta = std.math.pi / 2.0; // 90 degrees
    const phi = std.math.pi / -2.0;

    const camera = rl.Camera3D{
        .position = .{
            .x = center[0] + distance * std.math.sin(phi) * std.math.cos(theta),
            .y = center[1] + distance * std.math.cos(phi),
            .z = center[2] + distance * std.math.sin(phi) * std.math.sin(theta),
        },
        .target = .{ .x = center[0], .y = center[1], .z = center[2] },
        .up = .{ .x = 0, .y = -1, .z = 0 },
        .fovy = 45,
        .projection = rl.CameraProjection.perspective,
    };

    const cam_state = CamState{
        .distance = distance,
        .theta = theta,
        .phi = phi,
        .initial_theta = theta,
        .initial_phi = phi,
        .dragging = false,
        .mouse_start = .{ .x = 0, .y = 0 },
        .theta_start = theta,
        .phi_start = phi,
    };

    return CameraInitResult{
        .camera = camera,
        .cam_state = cam_state,
    };
}

const Chunk = struct {
    model: rl.Model,

    pub fn deinit(self: Chunk) void {
        rl.unloadModel(self.model);
    }
};

pub const GameState = struct {
    pub const config = .{
        .width = WIDTH,
        .height = HEIGHT,
        .title = "Gaussian Splat Viewer (Optimized)",
        .target_fps = 60,
    };
    camera: rl.Camera3D,
    cam_state: CamState,
    center: [3]f32,
    radius: f32,
    splat_data: []u8,
    vertex_count: usize,
    splats: []Splat,
    skip_factor: usize = 10,
    buf: [64]u8 = undefined,
    allocator: std.mem.Allocator,
    file_paths: std.ArrayListUnmanaged([]const u8),
    current_file_idx: usize,
    chunks: std.ArrayListUnmanaged(Chunk),
    shader: rl.Shader,

    pub fn init() !GameState {
        const allocator = std.heap.page_allocator;
        var file_paths = std.ArrayListUnmanaged([]const u8){};

        var dir = try std.fs.cwd().openDir("assets", .{ .iterate = true });
        defer dir.close();

        var it = dir.iterate();
        while (try it.next()) |entry| {
            if (entry.kind == .file and std.mem.endsWith(u8, entry.name, ".ply")) {
                const path = try std.fs.path.join(allocator, &[_][]const u8{ "assets", entry.name });
                try file_paths.append(allocator, path);
            }
        }

        if (file_paths.items.len == 0) return error.NoPlyFilesFound;

        const current_file_idx = 0;
        const result = try loadPly(allocator, file_paths.items[current_file_idx]);
        const center: [3]f32 = [_]f32{ 0, 0, 0 };
        const cam = initCamera(center);

        const shader = try rl.loadShader(null, "assets/splat.fs");

        var state = GameState{
            .camera = cam.camera,
            .cam_state = cam.cam_state,
            .center = center,
            .radius = 10.0,
            .splat_data = result.ply_data,
            .vertex_count = result.vertex_count,
            .splats = result.splats,
            .allocator = allocator,
            .file_paths = file_paths,
            .current_file_idx = current_file_idx,
            .chunks = .{},
            .shader = shader,
        };

        try state.rebuildChunks();

        return state;
    }

    pub fn deinit(self: *GameState) void {
        rl.unloadShader(self.shader);
        const allocator = self.allocator;
        for (self.chunks.items) |c| c.deinit();
        self.chunks.deinit(allocator);
        allocator.free(self.splats);
        allocator.free(self.splat_data);
        for (self.file_paths.items) |path| {
            allocator.free(path);
        }
        self.file_paths.deinit(allocator);
    }

    fn rebuildChunks(self: *GameState) !void {
        // Clear old chunks
        for (self.chunks.items) |c| c.deinit();
        self.chunks.clearRetainingCapacity();

        if (self.splats.len == 0) return;

        const SPLATS_PER_CHUNK = 60000;
        var processed: usize = 0;

        std.debug.print("Building chunks with skip_factor={}...\n", .{self.skip_factor});

        while (processed < self.splats.len) {
            const end = @min(processed + SPLATS_PER_CHUNK, self.splats.len);

            // Count actual vertices needed
            var count: usize = 0;
            for (processed..end) |i| {
                if (i % self.skip_factor == 0) count += 1;
            }

            if (count > 0) {
                var mesh = std.mem.zeroes(rl.Mesh);
                // 2 triangles (6 vertices) per splat for a quad
                mesh.vertexCount = @intCast(count * 6);
                mesh.triangleCount = @intCast(count * 2);

                const vertices = try self.allocator.alloc(f32, @as(usize, @intCast(mesh.vertexCount)) * 3);
                defer self.allocator.free(vertices);
                const texcoords = try self.allocator.alloc(f32, @as(usize, @intCast(mesh.vertexCount)) * 2);
                defer self.allocator.free(texcoords);
                const colors = try self.allocator.alloc(u8, @as(usize, @intCast(mesh.vertexCount)) * 4);
                defer self.allocator.free(colors);

                var v_idx: usize = 0;
                var t_idx: usize = 0;
                var c_idx: usize = 0;

                for (processed..end) |i| {
                    if (i % self.skip_factor != 0) continue;

                    const s = self.splats[i];
                    const x = s.pos[0];
                    const y = s.pos[1];
                    const z = s.pos[2];
                    const sz = 0.01;

                    // V1 (BL)
                    vertices[v_idx] = x - sz;
                    vertices[v_idx + 1] = y - sz;
                    vertices[v_idx + 2] = z;
                    texcoords[t_idx] = 0.0;
                    texcoords[t_idx + 1] = 0.0;
                    // V2 (TL)
                    vertices[v_idx + 3] = x - sz;
                    vertices[v_idx + 4] = y + sz;
                    vertices[v_idx + 5] = z;
                    texcoords[t_idx + 2] = 0.0;
                    texcoords[t_idx + 3] = 1.0;
                    // V3 (TR)
                    vertices[v_idx + 6] = x + sz;
                    vertices[v_idx + 7] = y + sz;
                    vertices[v_idx + 8] = z;
                    texcoords[t_idx + 4] = 1.0;
                    texcoords[t_idx + 5] = 1.0;

                    // V4 (BL)
                    vertices[v_idx + 9] = x - sz;
                    vertices[v_idx + 10] = y - sz;
                    vertices[v_idx + 11] = z;
                    texcoords[t_idx + 6] = 0.0;
                    texcoords[t_idx + 7] = 0.0;
                    // V5 (TR)
                    vertices[v_idx + 12] = x + sz;
                    vertices[v_idx + 13] = y + sz;
                    vertices[v_idx + 14] = z;
                    texcoords[t_idx + 8] = 1.0;
                    texcoords[t_idx + 9] = 1.0;
                    // V6 (BR)
                    vertices[v_idx + 15] = x + sz;
                    vertices[v_idx + 16] = y - sz;
                    vertices[v_idx + 17] = z;
                    texcoords[t_idx + 10] = 1.0;
                    texcoords[t_idx + 11] = 0.0;

                    v_idx += 18;
                    t_idx += 12;

                    const r = s.r;
                    const g = s.g;
                    const b = s.b;
                    const a = s.a;
                    for (0..6) |_| {
                        colors[c_idx] = r;
                        colors[c_idx + 1] = g;
                        colors[c_idx + 2] = b;
                        colors[c_idx + 3] = a;
                        c_idx += 4;
                    }
                }

                mesh.vertices = vertices.ptr;
                mesh.texcoords = texcoords.ptr;
                mesh.colors = colors.ptr;

                rl.uploadMesh(&mesh, false);

                // Clear pointers so Raylib doesn't try to free our allocator memory
                mesh.vertices = null;
                mesh.texcoords = null;
                mesh.colors = null;

                const model = try rl.loadModelFromMesh(mesh);
                model.materials[0].shader = self.shader;
                try self.chunks.append(self.allocator, Chunk{ .model = model });
            }
            processed = end;
        }
        std.debug.print("Created {} chunks.\n", .{self.chunks.items.len});
    }

    pub fn update(self: *GameState, dt: f32) void {
        if (rl.isKeyPressed(rl.KeyboardKey.space)) {
            const next_idx = (self.current_file_idx + 1) % self.file_paths.items.len;
            const next_path = self.file_paths.items[next_idx];
            std.debug.print("Loading {s}...\n", .{next_path});

            if (loadPly(self.allocator, next_path)) |res| {
                self.allocator.free(self.splats);
                self.allocator.free(self.splat_data);

                self.splats = res.splats;
                self.splat_data = res.ply_data;
                self.vertex_count = res.vertex_count;
                self.current_file_idx = next_idx;

                self.rebuildChunks() catch |err| {
                    std.debug.print("Failed to rebuild chunks: {}\n", .{err});
                };
            } else |err| {
                std.debug.print("Failed to load {s}: {}\n", .{ next_path, err });
            }
        }

        // Wheel for distance
        const wheel = rl.getMouseWheelMove();
        if (wheel != 0) {
            self.cam_state.distance *= std.math.pow(f32, 0.9, wheel);
            self.cam_state.distance = std.math.clamp(self.cam_state.distance, 0.1, 4.0);
        }

        // Mouse drag for rotation
        if (rl.isMouseButtonPressed(rl.MouseButton.left)) {
            self.cam_state.dragging = true;
            self.cam_state.mouse_start = rl.getMousePosition();
            self.cam_state.theta_start = self.cam_state.theta;
            self.cam_state.phi_start = self.cam_state.phi;
        }

        if (rl.isMouseButtonReleased(rl.MouseButton.left)) {
            self.cam_state.dragging = false;
        }

        if (self.cam_state.dragging and rl.isMouseButtonDown(rl.MouseButton.left)) {
            const current_pos = rl.getMousePosition();
            const delta_x = current_pos.x - self.cam_state.mouse_start.x;
            const delta_y = current_pos.y - self.cam_state.mouse_start.y;
            const sensitivity: f32 = 0.001;
            self.cam_state.theta = self.cam_state.theta_start + delta_x * sensitivity;
            self.cam_state.phi = self.cam_state.phi_start + delta_y * sensitivity;
            self.cam_state.theta = std.math.clamp(self.cam_state.theta, -std.math.pi, std.math.pi);
            self.cam_state.phi = std.math.clamp(self.cam_state.phi, -std.math.pi / 2.0, std.math.pi / 2.0);
            const delta_rad: f32 = 20.0 * std.math.pi / 180.0;
            self.cam_state.theta = std.math.clamp(self.cam_state.theta, self.cam_state.initial_theta - delta_rad, self.cam_state.initial_theta + delta_rad);
            self.cam_state.phi = std.math.clamp(self.cam_state.phi, self.cam_state.initial_phi - delta_rad, self.cam_state.initial_phi + delta_rad);
        }

        // Key bindings for skip factor
        const old_skip = self.skip_factor;
        if (rl.isKeyPressed(rl.KeyboardKey.one)) self.skip_factor = 1;
        if (rl.isKeyPressed(rl.KeyboardKey.two)) self.skip_factor = 10;
        if (rl.isKeyPressed(rl.KeyboardKey.three)) self.skip_factor = 50;
        if (rl.isKeyPressed(rl.KeyboardKey.four)) self.skip_factor = 100;

        if (self.skip_factor != old_skip) {
            self.rebuildChunks() catch |err| {
                std.debug.print("Failed to rebuild chunks: {}\n", .{err});
            };
        }

        // Vertigo effect (Dolly Zoom)
        if (rl.isKeyDown(rl.KeyboardKey.q) or rl.isKeyDown(rl.KeyboardKey.w)) {
            const zoom_speed = 30.0 * dt;
            const current_fov_rad = self.camera.fovy * std.math.pi / 180.0;
            const view_height = 2.0 * self.cam_state.distance * std.math.tan(current_fov_rad / 2.0);

            if (rl.isKeyDown(rl.KeyboardKey.w)) {
                self.camera.fovy += zoom_speed;
            }
            if (rl.isKeyDown(rl.KeyboardKey.q)) {
                self.camera.fovy -= zoom_speed;
            }

            self.camera.fovy = std.math.clamp(self.camera.fovy, 1.0, 110.0);

            const new_fov_rad = self.camera.fovy * std.math.pi / 180.0;
            self.cam_state.distance = view_height / (2.0 * std.math.tan(new_fov_rad / 2.0));
        }

        // Move camera target (E/R)
        if (rl.isKeyDown(rl.KeyboardKey.e) or rl.isKeyDown(rl.KeyboardKey.r)) {
            const move_speed = 5.0 * dt;

            // Move along Z axis only (World Space)
            if (rl.isKeyDown(rl.KeyboardKey.e)) {
                self.center[2] -= move_speed;
            }
            if (rl.isKeyDown(rl.KeyboardKey.r)) {
                self.center[2] += move_speed;
            }
            self.center[2] = std.math.clamp(self.center[2], -10.0, 10.0);
            self.camera.target = .{ .x = self.center[0], .y = self.center[1], .z = self.center[2] };
        }

        // Update camera position
        const cx = self.cam_state.distance * std.math.sin(self.cam_state.phi) * std.math.cos(self.cam_state.theta);
        const cy = self.cam_state.distance * std.math.cos(self.cam_state.phi);
        const cz = self.cam_state.distance * std.math.sin(self.cam_state.phi) * std.math.sin(self.cam_state.theta);
        self.camera.position = .{ .x = self.center[0] + cx, .y = self.center[1] + cy, .z = self.center[2] + cz };
    }

    pub fn render(self: *GameState) void {
        rl.beginDrawing();
        defer rl.endDrawing();
        rl.clearBackground(rl.Color.dark_blue);

        rl.beginMode3D(self.camera);

        rl.gl.rlDisableDepthMask();
        // Draw chunks
        const pos = rl.Vector3{ .x = 0, .y = 0, .z = 0 };
        for (self.chunks.items) |chunk| {
            rl.drawModel(chunk.model, pos, 1.0, rl.Color.white);
        }
        rl.gl.rlEnableDepthMask();

        rl.endMode3D();

        rl.drawFPS(10, 10);

        const num_rendered = if (self.splats.len == 0) 0 else ((self.splats.len - 1) / self.skip_factor) + 1;
        _ = std.fmt.bufPrintZ(&self.buf, "Rendered points: {}", .{num_rendered}) catch "Error";
        rl.drawText(@ptrCast(&self.buf), 10, 30, 20, rl.Color.white);
    }
};

pub fn main() !void {
    try Engine.run(GameState);
}
