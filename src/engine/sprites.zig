const rl = @import("raylib");
const std = @import("std");
const tl = @import("../game/texture_list.zig").TextureList;

pub const Sprites = struct {
    textures: [tl.count]rl.Texture = undefined,

    pub fn load() !Sprites {
        var s: Sprites = .{};
        inline for (tl.paths, 0..) |path, i| {
            s.textures[i] = try rl.loadTexture(path);
        }
        return s;
    }

    pub fn get(self: Sprites, id: tl.Id) rl.Texture {
        return self.textures[@intFromEnum(id)];
    }

    pub fn deinit(self: *Sprites) void {
        for (&self.textures) |*tex| {
            if (tex.id != 0) {
                rl.unloadTexture(tex.*);
            }
        }
    }
};
