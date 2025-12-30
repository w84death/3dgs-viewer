const std = @import("std");

pub const Vec2 = struct {
    x: f32,
    y: f32,

    pub fn init(x: f32, y: f32) Vec2 {
        return .{ .x = x, .y = y };
    }
    pub fn add(a: Vec2, b: Vec2) Vec2 {
        return .{ .x = a.x + b.x, .y = a.y + b.y };
    }
    pub fn sub(a: Vec2, b: Vec2) Vec2 {
        return .{ .x = a.x - b.x, .y = a.y - b.y };
    }
    pub fn scale(v: Vec2, s: f32) Vec2 {
        return .{ .x = v.x * s, .y = v.y * s };
    }
    pub fn clamp(v: Vec2, min: f32, max_val: f32) Vec2 {
        return .{ .x = std.math.clamp(v.x, min, max_val), .y = std.math.clamp(v.y, min, max_val) };
    }
};

pub const IVec2 = struct {
    x: i32,
    y: i32,
    pub fn init(x: i32, y: i32) IVec2 {
        return .{ .x = x, .y = y };
    }
};

pub fn max(a: f32, b: f32) f32 {
    return if (a > b) a else b;
}
