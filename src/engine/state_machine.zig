const std = @import("std");
const rl = @import("raylib");

pub const GameStateType = enum {
    Intro,
    Menu,
    Game,
    Credits,
    Exit,
};

pub const StateCallbacks = struct {
    updateIntro: *const fn (ctx: *anyopaque, dt: f32) void,
    updateMenu: *const fn (ctx: *anyopaque, dt: f32) void,
    updateGame: *const fn (ctx: *anyopaque, dt: f32) void,
    updateCredits: *const fn (ctx: *anyopaque, dt: f32) void,

    // Render functions
    renderIntro: *const fn (ctx: *anyopaque) void,
    renderMenu: *const fn (ctx: *anyopaque) void,
    renderGame: *const fn (ctx: *anyopaque) void,
    renderCredits: *const fn (ctx: *anyopaque) void,

    // Lifecycle functions
    enterState: *const fn (ctx: *anyopaque, state: GameStateType) void,
    exitState: *const fn (ctx: *anyopaque, state: GameStateType) void,
};

pub const StateMachine = struct {
    current_state: GameStateType = .Intro,
    next_state: ?GameStateType = null,
    game_context: *anyopaque,
    callbacks: StateCallbacks,

    pub fn init(game_context: *anyopaque, callbacks: StateCallbacks) StateMachine {
        return .{
            .game_context = game_context,
            .callbacks = callbacks,
        };
    }

    pub fn switchTo(self: *StateMachine, new_state: GameStateType) void {
        self.next_state = new_state;
    }

    pub fn update(self: *StateMachine, dt: f32) void {
        if (self.next_state) {
            self.callbacks.exitState(self.game_context, self.current_state);
            if (self.next_state) |next| {
                self.current_state = next;
                self.next_state = null;
                self.callbacks.enterState(self.game_context, self.current_state);
            }
        }

        switch (self.current_state) {
            .Intro => self.callbacks.updateIntro(self.game_context, dt),
            .Menu => self.callbacks.updateMenu(self.game_context, dt),
            .Game => self.callbacks.updateGame(self.game_context, dt),
            .Credits => self.callbacks.updateCredits(self.game_context, dt),
            .Exit => rl.closeWindow(),
        }
    }

    pub fn render(self: *StateMachine) void {
        switch (self.current_state) {
            .Intro => self.callbacks.renderIntro(self.game_context),
            .Menu => self.callbacks.renderMenu(self.game_context),
            .Game => self.callbacks.renderGame(self.game_context),
            .Credits => self.callbacks.renderCredits(self.game_context),
            .Exit => {},
        }
    }
};
