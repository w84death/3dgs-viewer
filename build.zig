const std = @import("std");
pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const raylib_dep = b.dependency("raylib_zig", .{
        .target = target,
        .optimize = optimize,
    });
    const exe = b.addExecutable(.{
        .name = "gaussian_splat_viewer",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const raylib_artifact = raylib_dep.artifact("raylib");
    exe.linkLibrary(raylib_artifact);

    const raylib = raylib_dep.module("raylib");
    exe.root_module.addImport("raylib", raylib);

    b.installArtifact(exe);

    const run_step = b.step("run", "Run the app");
    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| run_cmd.addArgs(args);

    const upx_step = b.step("upx", "Compress binary");
    const install_path = b.getInstallPath(.bin, "gaussian_splat_viewer");
    const compress = b.addSystemCommand(&[_][]const u8{
        "upx",
        "--best",
        "--lzma",
        "--compress-icons=0",
        install_path,
    });
    compress.step.dependOn(b.getInstallStep());
    upx_step.dependOn(&compress.step);
}
