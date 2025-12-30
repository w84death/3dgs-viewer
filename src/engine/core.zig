const rl = @import("raylib");
const std = @import("std");

pub const Config = struct {
    width: i32 = 800,
    height: i32 = 600,
    title: [:0]const u8 = "Raylib/Zig Engine",
    target_fps: i32 = 60,
};

pub const Engine = struct {
    pub fn run(comptime GameState: type) !void {
        const config = GameState.config;

        rl.initWindow(config.width, config.height, config.title);
        defer rl.closeWindow();
        rl.setTargetFPS(config.target_fps);

        rl.initAudioDevice();
        defer rl.closeAudioDevice();

        var game_state = try GameState.init();
        defer game_state.deinit();

        while (!rl.windowShouldClose()) {
            const dt = rl.getFrameTime();
            game_state.update(dt);
            game_state.render();
        }
    }
};
