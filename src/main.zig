const rl = @import("raylib");
const std = @import("std");
const Thread = std.Thread;
const Atomic = std.atomic.Value(bool);
const WIDTH = 1280;
const HEIGHT = 800;
const MOUSE_SENSITIVITY = 0.001;
const CAMERA_DELTA_RAD = 10.0 * std.math.pi / 180.0;
const CAMERA_PHI_MIN = 0.001;
const CAMERA_PHI_MAX = std.math.pi - 0.001;
const ZOOM_SPEED_BASE = 20.0;
const MOVE_SPEED_BASE = 2.0;
const FOV_MIN = 2.0;
const FOV_MAX = 90.0;
const CENTER_Z_MIN = -1.0;
const CENTER_Z_MAX = 2.0;
const ROTATION_STOP_CAP = 16;
const SORT_TRIGGER_FRAME = 15;
const SORT_CHECK_INTERVAL = 30;

const UI = struct {
    const FPS_POS_X = 10;
    const FPS_POS_Y = 10;
    const POINTS_POS_X = 10;
    const POINTS_POS_Y = 30;
    const POINTS_FONT_SIZE = 20;
    const LOADING_RECT_X = 10;
    const LOADING_RECT_Y = 40;
    const LOADING_RECT_W = 100;
    const LOADING_RECT_H = 25;
    const LOADING_TEXT_X = 15;
    const LOADING_TEXT_Y = 45;
    const LOADING_FONT_SIZE = 16;
};

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
    scale: [3]f32,
    r: u8,
    g: u8,
    b: u8,
    a: u8,
};

const PlyLoadResult = struct {
    splats: []Splat,
    vertex_count: usize,
};

fn loadPly(allocator: std.mem.Allocator, path: []const u8) !PlyLoadResult {
    const ply_data = try std.fs.cwd().readFileAlloc(allocator, path, std.math.maxInt(usize));
    defer allocator.free(ply_data);

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
        var scale: [3]f32 = undefined;
        var color_f: [3]f32 = undefined;
        var opacity: f32 = 0;
        for (properties.items, 0..) |name, idx| {
            const val = f32_slice[off + idx];
            if (std.mem.eql(u8, name, "x")) pos[0] = val else if (std.mem.eql(u8, name, "y")) pos[1] = val else if (std.mem.eql(u8, name, "z")) pos[2] = val else if (std.mem.eql(u8, name, "scale_0")) scale[0] = val else if (std.mem.eql(u8, name, "scale_1")) scale[1] = val else if (std.mem.eql(u8, name, "scale_2")) scale[2] = val else if (std.mem.eql(u8, name, "f_dc_0")) color_f[0] = val else if (std.mem.eql(u8, name, "f_dc_1")) color_f[1] = val else if (std.mem.eql(u8, name, "f_dc_2")) color_f[2] = val else if (std.mem.eql(u8, name, "opacity")) opacity = val;
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
            .scale = .{ std.math.exp(scale[0]), std.math.exp(scale[1]), std.math.exp(scale[2]) },
            .r = r,
            .g = g,
            .b = b,
            .a = a,
        };
        if (ii % 100000 == 0 or ii == vertex_count - 1) std.debug.print("Loaded {}/{} vertices\n", .{ ii + 1, vertex_count });
    }
    std.debug.print("Finished loading {} vertices from binary PLY\n", .{vertex_count});

    return PlyLoadResult{
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
    const theta = -std.math.pi / 2.0;
    const phi = std.math.pi / 2.0;

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

const SPLAT_VS =
    \\#version 330
    \\
    \\in vec3 vertexPosition;
    \\in vec2 vertexTexCoord;
    \\in vec3 vertexNormal;
    \\in vec4 vertexColor;
    \\
    \\out vec2 fragTexCoord;
    \\out vec4 fragColor;
    \\
    \\uniform mat4 mvp;
    \\uniform mat4 matView;
    \\
    \\void main()
    \\{
    \\    fragTexCoord = vertexTexCoord;
    \\    fragColor = vertexColor;
    \\
    \\    vec3 center = vertexPosition;
    \\    vec2 size = vertexNormal.xy * 2.0;
    \\
    \\    vec3 right = vec3(matView[0][0], matView[1][0], matView[2][0]);
    \\    vec3 up    = vec3(matView[0][1], matView[1][1], matView[2][1]);
    \\
    \\    vec3 pos = center
    \\        + (right * (vertexTexCoord.x - 0.5) * size.x)
    \\        + (up    * (vertexTexCoord.y - 0.5) * size.y);
    \\
    \\    gl_Position = mvp * vec4(pos, 1.0);
    \\}
;

const SPLAT_FS =
    \\#version 330
    \\
    \\in vec2 fragTexCoord;
    \\in vec4 fragColor;
    \\out vec4 color;
    \\
    \\void main()
    \\{
    \\    vec2 center = vec2(0.5, 0.5);
    \\    vec2 p = fragTexCoord - center;
    \\    float r2 = dot(p, p);
    \\
    \\    if (r2 > 0.5) discard;
    \\    color.rgb = pow(color.rgb, vec3(1.0/2.2));
    \\
    \\    float alpha = exp(-r2 * 10.0);
    \\    color = fragColor;
    \\    color.rgb = pow(color.rgb, vec3(1.0/2.2));
    \\    color.a *= alpha;
    \\}
;

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
    vertex_count: usize,
    rendered_splats_count: usize = 0,
    splats: []Splat,
    skip_factor: usize = 1,
    buf: [64]u8 = undefined,
    allocator: std.mem.Allocator,
    file_paths: []const []const u8,
    current_file_idx: usize,
    chunks: std.ArrayListUnmanaged(Chunk),
    shader: rl.Shader,

    // Scratch buffers for chunk generation
    scratch_vertices: []f32,
    scratch_texcoords: []f32,
    scratch_normals: []f32,
    scratch_colors: []u8,

    frame_count: usize = 0,
    needs_sort: bool = false,
    background_splats: []Splat,
    sort_thread: ?Thread = null,
    sort_done: Atomic = Atomic.init(false),
    rotation_stop_counter: usize = 0,
    is_loading: bool = false,

    const SPLATS_PER_CHUNK = 60000;

    pub fn initWithIdx(allocator: std.mem.Allocator, file_paths: []const []const u8, file_idx: usize) !GameState {
        if (file_idx >= file_paths.len) return error.InvalidIndex;

        const result = try loadPly(allocator, file_paths[file_idx]);
        const center: [3]f32 = [_]f32{ 0, 0, 0 };
        const cam = initCamera(center);

        var shader = try rl.loadShaderFromMemory(SPLAT_VS, SPLAT_FS);
        shader.locs[@intFromEnum(rl.ShaderLocationIndex.matrix_view)] = rl.getShaderLocation(shader, "matView");

        var self = GameState{
            .background_splats = &[_]Splat{},
            .camera = cam.camera,
            .cam_state = cam.cam_state,
            .center = center,
            .radius = 10.0,
            .vertex_count = result.vertex_count,
            .splats = result.splats,
            .skip_factor = 1,
            .allocator = allocator,
            .file_paths = file_paths,
            .current_file_idx = file_idx,
            .chunks = try std.ArrayList(Chunk).initCapacity(allocator, 0),
            .shader = shader,
            .scratch_vertices = try allocator.alloc(f32, SPLATS_PER_CHUNK * 6 * 3),
            .scratch_texcoords = try allocator.alloc(f32, SPLATS_PER_CHUNK * 6 * 2),
            .scratch_normals = try allocator.alloc(f32, SPLATS_PER_CHUNK * 6 * 3),
            .scratch_colors = try allocator.alloc(u8, SPLATS_PER_CHUNK * 6 * 4),
        };

        self.background_splats = self.allocator.alloc(Splat, self.vertex_count) catch unreachable;

        self.sortSplats();
        try self.rebuildChunks();

        return self;
    }

    pub fn deinit(self: *GameState) void {
        rl.unloadShader(self.shader);
        const allocator = self.allocator;
        for (self.chunks.items) |c| c.deinit();
        self.chunks.deinit(allocator);
        allocator.free(self.splats);
        allocator.free(self.scratch_vertices);
        allocator.free(self.scratch_texcoords);
        allocator.free(self.scratch_normals);
        allocator.free(self.scratch_colors);
        self.chunks.deinit(self.allocator);
    }

    fn sortSplats(self: *GameState) void {
        const allocator = self.allocator;
        const len = self.splats.len;
        if (len == 0) return;

        var dists = allocator.alloc(f32, len) catch unreachable;
        defer allocator.free(dists);

        const cam_pos = .{ self.camera.position.x, self.camera.position.y, self.camera.position.z };

        // SIMD compute squared distances for batches of 4
        const Vec4 = @Vector(4, f32);
        var i: usize = 0;
        while (i + 3 < len) {
            const pos0 = self.splats[i].pos;
            const pos1 = self.splats[i + 1].pos;
            const pos2 = self.splats[i + 2].pos;
            const pos3 = self.splats[i + 3].pos;

            const dx: Vec4 = .{ pos0[0] - cam_pos[0], pos1[0] - cam_pos[0], pos2[0] - cam_pos[0], pos3[0] - cam_pos[0] };
            const dy: Vec4 = .{ pos0[1] - cam_pos[1], pos1[1] - cam_pos[1], pos2[1] - cam_pos[1], pos3[1] - cam_pos[1] };
            const dz: Vec4 = .{ pos0[2] - cam_pos[2], pos1[2] - cam_pos[2], pos2[2] - cam_pos[2], pos3[2] - cam_pos[2] };

            const dist_sq: Vec4 = dx * dx + dy * dy + dz * dz;
            const dist_array = @as([4]f32, dist_sq);
            std.mem.copyForwards(f32, dists[i .. i + 4], &dist_array);
            i += 4;
        }

        // Scalar for remainder
        while (i < len) {
            const pos = self.splats[i].pos;
            const dx = pos[0] - cam_pos[0];
            const dy = pos[1] - cam_pos[1];
            const dz = pos[2] - cam_pos[2];
            dists[i] = dx * dx + dy * dy + dz * dz;
            i += 1;
        }

        // Sort indices based on dists (smaller dist_sq first for front-to-back)
        var indices = allocator.alloc(usize, len) catch unreachable;
        defer allocator.free(indices);
        for (0..len) |j| indices[j] = j;

        const SortContext = struct {
            dists: []f32,
            pub fn lessThan(ctx: @This(), a: usize, b: usize) bool {
                return ctx.dists[a] > ctx.dists[b];
            }
        };

        const ctx = SortContext{ .dists = dists };
        std.sort.block(usize, indices, ctx, SortContext.lessThan);

        // Reorder splats based on sorted indices
        var temp = allocator.alloc(Splat, len) catch unreachable;
        defer allocator.free(temp);
        for (0..len) |j| {
            temp[j] = self.splats[indices[j]];
        }
        @memcpy(self.splats, temp);
    }

    fn rebuildChunks(self: *GameState) !void {
        for (self.chunks.items) |c| c.deinit();
        self.chunks.clearRetainingCapacity();

        if (self.splats.len == 0) return;

        var processed: usize = 0;

        while (processed < self.splats.len) {
            const end = @min(processed + SPLATS_PER_CHUNK, self.splats.len);
            var count: usize = 0;
            var start_idx = processed;
            if (start_idx % self.skip_factor != 0) {
                start_idx += self.skip_factor - (start_idx % self.skip_factor);
            }

            if (start_idx < end) {
                count = (end - 1 - start_idx) / self.skip_factor + 1;
            }

            if (count > 0) {
                var mesh = std.mem.zeroes(rl.Mesh);
                mesh.vertexCount = @intCast(count * 6);
                mesh.triangleCount = @intCast(count * 2);

                const vertex_float_count = @as(usize, @intCast(mesh.vertexCount)) * 3;
                const tex_float_count = @as(usize, @intCast(mesh.vertexCount)) * 2;
                const normal_float_count = @as(usize, @intCast(mesh.vertexCount)) * 3;
                const color_byte_count = @as(usize, @intCast(mesh.vertexCount)) * 4;

                const vertices = self.scratch_vertices[0..vertex_float_count];
                const texcoords = self.scratch_texcoords[0..tex_float_count];
                const normals = self.scratch_normals[0..normal_float_count];
                const colors = self.scratch_colors[0..color_byte_count];

                var v_idx: usize = 0;
                var t_idx: usize = 0;
                var c_idx: usize = 0;

                const Vec3 = @Vector(3, f32);

                var i = start_idx;
                while (i < end) : (i += self.skip_factor) {
                    const s = self.splats[i];
                    const x = s.pos[0];
                    const y = s.pos[1];
                    const z = s.pos[2];
                    const sx = 2.0 * s.scale[0];
                    const sy = 2.0 * s.scale[1];

                    // SIMD-optimized vertex and normal filling
                    const pos_vec: Vec3 = .{ x, y, z };
                    const normal_vec: Vec3 = .{ sx, sy, 0.0 };
                    for (0..6) |k| {
                        std.mem.copyForwards(f32, vertices[v_idx + k * 3 .. v_idx + k * 3 + 3], &@as([3]f32, pos_vec));
                        std.mem.copyForwards(f32, normals[v_idx + k * 3 .. v_idx + k * 3 + 3], &@as([3]f32, normal_vec));
                    }

                    // SIMD-optimized texcoord filling
                    const tex_vec: @Vector(12, f32) = .{ 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0 };
                    std.mem.copyForwards(f32, texcoords[t_idx .. t_idx + 12], &@as([12]f32, tex_vec));

                    v_idx += 18;
                    t_idx += 12;

                    const r = s.r;
                    const g = s.g;
                    const b = s.b;
                    const a = s.a;
                    // SIMD-optimized color filling
                    const color_vec: @Vector(4, u8) = .{ r, g, b, a };
                    for (0..6) |_| {
                        std.mem.copyForwards(u8, colors[c_idx .. c_idx + 4], &@as([4]u8, color_vec));
                        c_idx += 4;
                    }
                }

                mesh.vertices = vertices.ptr;
                mesh.texcoords = texcoords.ptr;
                mesh.normals = normals.ptr;
                mesh.colors = colors.ptr;

                rl.uploadMesh(&mesh, false);

                // Clear pointers so Raylib doesn't try to free our allocator memory
                mesh.vertices = null;
                mesh.texcoords = null;
                mesh.normals = null;
                mesh.colors = null;

                const model = try rl.loadModelFromMesh(mesh);
                model.materials[0].shader = self.shader;
                try self.chunks.append(self.allocator, Chunk{ .model = model });
            }
            processed = end;
        }

        var total_rendered: usize = 0;
        for (self.chunks.items) |chunk| {
            total_rendered += @intCast(@divFloor(chunk.model.meshes[0].vertexCount, 6));
        }
        self.rendered_splats_count = total_rendered;
    }

    // Handle user input (keys and mouse)
    fn handleInput(self: *GameState) void {
        if (rl.isKeyPressed(rl.KeyboardKey.f11)) {
            rl.toggleFullscreen();
        }

        if (rl.isKeyPressed(rl.KeyboardKey.space)) {
            self.is_loading = true;
        }

        if (rl.isMouseButtonPressed(rl.MouseButton.left)) {
            self.cam_state.dragging = true;
            self.cam_state.mouse_start = rl.getMousePosition();
            self.cam_state.theta_start = self.cam_state.theta;
            self.cam_state.phi_start = self.cam_state.phi;
        }

        if (rl.isMouseButtonReleased(rl.MouseButton.left)) {
            self.cam_state.dragging = false;
        }

        const old_skip = self.skip_factor;
        const skip_keys = [_]struct { key: rl.KeyboardKey, value: usize }{
            .{ .key = rl.KeyboardKey.one, .value = 1 },
            .{ .key = rl.KeyboardKey.two, .value = 2 },
            .{ .key = rl.KeyboardKey.three, .value = 5 },
            .{ .key = rl.KeyboardKey.four, .value = 10 },
            .{ .key = rl.KeyboardKey.five, .value = 25 },
        };
        for (skip_keys) |sk| {
            if (rl.isKeyPressed(sk.key)) self.skip_factor = sk.value;
        }
        if (rl.isKeyReleased(rl.KeyboardKey.f)) rl.toggleFullscreen();

        if (self.skip_factor != old_skip) {
            self.rebuildChunks() catch |err| {
                std.debug.print("Failed to rebuild chunks: {}\n", .{err});
            };
        }
    }

    // Update camera state based on input and time
    fn updateCamera(self: *GameState, dt: f32) void {
        if (self.cam_state.dragging and rl.isMouseButtonDown(rl.MouseButton.left)) {
            const current_pos = rl.getMousePosition();
            const delta_x = self.cam_state.mouse_start.x - current_pos.x;
            const delta_y = self.cam_state.mouse_start.y - current_pos.y;
            self.cam_state.theta = self.cam_state.theta_start - delta_x * MOUSE_SENSITIVITY;
            self.cam_state.phi = self.cam_state.phi_start + delta_y * MOUSE_SENSITIVITY;

            self.cam_state.theta = std.math.clamp(self.cam_state.theta, self.cam_state.initial_theta - CAMERA_DELTA_RAD, self.cam_state.initial_theta + CAMERA_DELTA_RAD);
            self.cam_state.phi = std.math.clamp(self.cam_state.phi, self.cam_state.initial_phi - CAMERA_DELTA_RAD, self.cam_state.initial_phi + CAMERA_DELTA_RAD);

            self.cam_state.phi = std.math.clamp(self.cam_state.phi, CAMERA_PHI_MIN, CAMERA_PHI_MAX);
        }

        // Vertigo effect (Dolly Zoom)
        var fov_delta: f32 = 0;
        if (rl.isKeyDown(rl.KeyboardKey.w)) fov_delta += ZOOM_SPEED_BASE * dt;
        if (rl.isKeyDown(rl.KeyboardKey.q)) fov_delta -= ZOOM_SPEED_BASE * dt;
        if (fov_delta != 0) {
            self.camera.fovy += fov_delta;
            self.camera.fovy = std.math.clamp(self.camera.fovy, FOV_MIN, FOV_MAX);
            const current_fov_rad = self.camera.fovy * std.math.pi / 180.0;
            const view_height = 2.0 * self.cam_state.distance * std.math.tan(current_fov_rad / 2.0);
            const new_fov_rad = self.camera.fovy * std.math.pi / 180.0;
            self.cam_state.distance = view_height / (2.0 * std.math.tan(new_fov_rad / 2.0));
        }

        var z_delta: f32 = 0;
        if (rl.isKeyDown(rl.KeyboardKey.up)) z_delta += MOVE_SPEED_BASE * dt;
        if (rl.isKeyDown(rl.KeyboardKey.down)) z_delta -= MOVE_SPEED_BASE * dt;
        if (z_delta != 0) {
            self.center[2] += z_delta;
            self.center[2] = std.math.clamp(self.center[2], CENTER_Z_MIN, CENTER_Z_MAX);
            self.camera.target = .{ .x = self.center[0], .y = self.center[1], .z = self.center[2] };
        }

        const cx = self.cam_state.distance * std.math.sin(self.cam_state.phi) * std.math.cos(self.cam_state.theta);
        const cy = self.cam_state.distance * std.math.cos(self.cam_state.phi);
        const cz = self.cam_state.distance * std.math.sin(self.cam_state.phi) * std.math.sin(self.cam_state.theta);
        self.camera.position = .{ .x = self.center[0] + cx, .y = self.center[1] + cy, .z = self.center[2] + cz };
    }

    // Manage sorting thread and related logic
    fn manageSorting(self: *GameState) void {
        if (self.cam_state.dragging) {
            self.rotation_stop_counter = 0;
        } else {
            self.rotation_stop_counter = @min(self.rotation_stop_counter + 1, ROTATION_STOP_CAP); // Cap to avoid overflow, 31 > 30
            if (self.rotation_stop_counter == SORT_TRIGGER_FRAME) {
                self.needs_sort = true;
            }
        }

        if (self.sort_thread) |*t| {
            if (self.sort_done.load(.acquire)) {
                t.join();
                self.sort_thread = null;

                // Swap buffers
                const temp = self.splats;
                self.splats = self.background_splats;
                self.background_splats = temp;

                self.rebuildChunks() catch |err| {
                    std.debug.print("Failed to rebuild chunks: {}\n", .{err});
                };
                self.needs_sort = false;
            }
        }

        if (self.needs_sort and self.frame_count % SORT_CHECK_INTERVAL == 0 and self.sort_thread == null) {
            // Copy to background and spawn sort thread
            if (self.background_splats.len != self.splats.len) {
                self.allocator.free(self.background_splats);
                self.background_splats = self.allocator.alloc(Splat, self.splats.len) catch unreachable;
            }
            @memcpy(self.background_splats, self.splats);
            const cam_pos = .{ self.camera.position.x, self.camera.position.y, self.camera.position.z };
            self.sort_done.store(false, .release);
            self.sort_thread = Thread.spawn(.{}, sortFunction, .{ self, cam_pos }) catch unreachable;
        }
    }

    pub fn update(self: *GameState, dt: f32) void {
        self.handleInput();

        self.frame_count += 1;

        if (self.is_loading) {
            const next_idx = (self.current_file_idx + 1) % self.file_paths.len;
            const next_path = self.file_paths[next_idx];
            std.debug.print("Loading {s}...\n", .{next_path});

            if (loadPly(self.allocator, next_path)) |res| {
                self.allocator.free(self.splats);
                if (self.background_splats.len > 0) self.allocator.free(self.background_splats);
                self.splats = res.splats;
                self.vertex_count = res.vertex_count;
                self.background_splats = self.allocator.alloc(Splat, self.vertex_count) catch unreachable;
                self.current_file_idx = next_idx;

                self.sortSplats();
                self.rebuildChunks() catch |err| {
                    std.debug.print("Failed to rebuild chunks: {}\n", .{err});
                };
            } else |err| {
                std.debug.print("Failed to load {s}: {}\n", .{ next_path, err });
            }
            self.is_loading = false;
        }

        self.updateCamera(dt);
        self.manageSorting();
    }

    pub fn render(self: *GameState) void {
        rl.beginDrawing();
        defer rl.endDrawing();
        rl.clearBackground(rl.Color.black);
        rl.drawCircleGradient(@divTrunc(rl.getScreenWidth(), 2), @divTrunc(rl.getScreenHeight(), 2), @floatFromInt(rl.getScreenHeight()), rl.Color.dark_blue, rl.Color.black);

        rl.beginMode3D(self.camera);

        rl.gl.rlDisableBackfaceCulling();
        rl.gl.rlDisableDepthMask();
        for (self.chunks.items) |chunk| {
            rl.drawMesh(chunk.model.meshes[0], chunk.model.materials[0], chunk.model.transform);
        }
        rl.gl.rlEnableDepthMask();
        rl.gl.rlEnableBackfaceCulling();

        rl.endMode3D();

        self.drawUI();
    }

    // Draw UI elements
    fn drawUI(self: *GameState) void {
        if (self.is_loading) {
            rl.drawRectangle(UI.LOADING_RECT_X, UI.LOADING_RECT_Y, UI.LOADING_RECT_W, UI.LOADING_RECT_H, rl.Color.black);
            rl.drawText("LOADING...", UI.LOADING_TEXT_X, UI.LOADING_TEXT_Y, UI.LOADING_FONT_SIZE, rl.Color.white);
        }

        rl.drawFPS(UI.FPS_POS_X, UI.FPS_POS_Y);

        _ = std.fmt.bufPrintZ(&self.buf, "Rendered points: {}", .{self.rendered_splats_count}) catch "Error";
        rl.drawText(@ptrCast(&self.buf), UI.POINTS_POS_X, UI.POINTS_POS_Y, UI.POINTS_FONT_SIZE, rl.Color.white);
    }
};

fn sortFunction(state: *GameState, cam_pos: [3]f32) void {
    const allocator = state.allocator;
    const len = state.background_splats.len;
    if (len == 0) {
        state.sort_done.store(true, .release);
        return;
    }

    var dists = allocator.alloc(f32, len) catch unreachable;
    defer allocator.free(dists);

    // SIMD compute squared distances for batches of 4
    const Vec4 = @Vector(4, f32);
    var i: usize = 0;
    while (i + 3 < len) {
        const pos0 = state.background_splats[i].pos;
        const pos1 = state.background_splats[i + 1].pos;
        const pos2 = state.background_splats[i + 2].pos;
        const pos3 = state.background_splats[i + 3].pos;

        const dx: Vec4 = .{ pos0[0] - cam_pos[0], pos1[0] - cam_pos[0], pos2[0] - cam_pos[0], pos3[0] - cam_pos[0] };
        const dy: Vec4 = .{ pos0[1] - cam_pos[1], pos1[1] - cam_pos[1], pos2[1] - cam_pos[1], pos3[1] - cam_pos[1] };
        const dz: Vec4 = .{ pos0[2] - cam_pos[2], pos1[2] - cam_pos[2], pos2[2] - cam_pos[2], pos3[2] - cam_pos[2] };

        const dist_sq: Vec4 = dx * dx + dy * dy + dz * dz;
        const dist_array = @as([4]f32, dist_sq);
        std.mem.copyForwards(f32, dists[i .. i + 4], &dist_array);
        i += 4;
    }

    // Scalar for remainder
    while (i < len) {
        const pos = state.background_splats[i].pos;
        const dx = pos[0] - cam_pos[0];
        const dy = pos[1] - cam_pos[1];
        const dz = pos[2] - cam_pos[2];
        dists[i] = dx * dx + dy * dy + dz * dz;
        i += 1;
    }

    // Sort indices based on dists (smaller dist_sq first for front-to-back)
    var indices = allocator.alloc(usize, len) catch unreachable;
    defer allocator.free(indices);
    for (0..len) |j| indices[j] = j;

    const SortContext = struct {
        dists: []f32,
        pub fn lessThan(ctx: @This(), a: usize, b: usize) bool {
            return ctx.dists[a] > ctx.dists[b];
        }
    };

    const ctx = SortContext{ .dists = dists };
    std.sort.block(usize, indices, ctx, SortContext.lessThan);

    // Reorder splats based on sorted indices
    var temp = allocator.alloc(Splat, len) catch unreachable;
    defer allocator.free(temp);
    for (0..len) |j| {
        temp[j] = state.background_splats[indices[j]];
    }
    @memcpy(state.background_splats, temp);

    state.sort_done.store(true, .release);
}

pub fn main() !void {
    const config = GameState.config;

    rl.initWindow(config.width, config.height, config.title);
    defer rl.closeWindow();
    rl.setTargetFPS(config.target_fps);

    var allocator = std.heap.page_allocator;
    var file_paths = std.ArrayListUnmanaged([]const u8){};
    defer {
        for (file_paths.items) |path| allocator.free(path);
        file_paths.deinit(allocator);
    }

    var text_buf: [256]u8 = undefined;

    var dir = try std.fs.cwd().openDir(".", .{ .iterate = true });
    defer dir.close();

    var it = dir.iterate();
    while (try it.next()) |entry| {
        if (entry.kind == .file and std.mem.endsWith(u8, entry.name, ".ply")) {
            const path = try allocator.dupe(u8, entry.name);
            try file_paths.append(allocator, path);
        }
    }

    if (file_paths.items.len == 0) return error.NoPlyFilesFound;

    var game_state: ?GameState = null;
    var selected_file: ?usize = null;

    while (!rl.windowShouldClose()) {
        const dt = rl.getFrameTime();

        if (game_state) |*gs| {
            gs.update(dt);
            gs.render();

            // Right-click to return to menu
            if (rl.isMouseButtonPressed(rl.MouseButton.right)) {
                gs.deinit();
                game_state = null;
                selected_file = null;
            }
        } else {
            // Menu: handle clicks
            if (rl.isMouseButtonPressed(rl.MouseButton.left)) {
                const mouse_pos = rl.getMousePosition();
                for (file_paths.items, 0..) |_, i| {
                    const rect_y = 90 + @as(i32, @intCast(i)) * 30;
                    if (mouse_pos.x >= 70 and mouse_pos.x <= 690 and mouse_pos.y >= @as(f32, @floatFromInt(rect_y - 2)) and mouse_pos.y <= @as(f32, @floatFromInt(rect_y + 28))) {
                        selected_file = i;
                        game_state = try GameState.initWithIdx(allocator, file_paths.items, i);
                        break;
                    }
                }
            }

            // Draw menu
            rl.beginDrawing();
            rl.clearBackground(rl.Color.black);
            rl.drawRectangle(50, 50, 700, 500, rl.Color.white);
            const title = std.fmt.bufPrintZ(&text_buf, "Select a PLY file to view:", .{}) catch "Error";
            rl.drawText(title, 60, 60, 24, rl.Color.black);
            for (file_paths.items, 0..) |path, i| {
                const y = 90 + @as(i32, @intCast(i)) * 30;
                const file_text = std.fmt.bufPrintZ(&text_buf, "{s}", .{path}) catch "Error";
                rl.drawRectangleLines(70, y - 2, 620, 30, rl.Color.black);
                rl.drawText(file_text, 80, y, 20, rl.Color.black);
            }
            const instr = std.fmt.bufPrintZ(&text_buf, "Left-click to select, Right-click does nothing", .{}) catch "Error";
            rl.drawText(instr, 60, 520, 18, rl.Color.gray);
            rl.endDrawing();
        }
    }
}
