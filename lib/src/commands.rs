use std::collections::HashMap;
use std::sync::Mutex;
use lazy_static::lazy_static;

pub struct Command {
    pub name: &'static str,
    pub func: fn(Vec<String>),
    pub help: &'static str,
    pub usage: &'static str,
}

lazy_static! {
    pub static ref COMMAND_REGISTRY: Mutex<HashMap<&'static str, Command>> = Mutex::new(HashMap::new());
}

pub fn register_command(command: Command) {
    COMMAND_REGISTRY.lock().unwrap().insert(command.name, command);
}

#[macro_export]
macro_rules! register_command {
    ($name:expr, $func:expr, $help:expr, $usage:expr) => {
        $crate::commands::register_command($crate::commands::Command {
            name: $name,
            func: $func,
            help: $help,
            usage: $usage,
        });
    };
}