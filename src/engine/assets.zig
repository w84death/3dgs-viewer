const rl = @import("raylib");
const TextureList = @import("../texture_list.zig").TextureList;

pub const AssetManager = struct {
    textures: [TextureList.count]rl.Texture,

    pub fn init() !AssetManager {
        var self: AssetManager = undefined;
        inline for (TextureList.paths, 0..) |path, i| {
            self.textures[i] = try rl.loadTexture(path);
        }
        return self;
    }

    pub fn deinit(self: *AssetManager) void {
        for (&self.textures) |*tex| {
            if (tex.id != 0) rl.unloadTexture(tex.*);
        }
    }

    pub fn get(self: AssetManager, id: TextureList.Id) rl.Texture {
        return self.textures[@intFromEnum(id)];
    }
};
