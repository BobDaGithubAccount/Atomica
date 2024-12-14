use std::collections::HashMap;
use std::sync::Mutex;
use lazy_static::lazy_static;
use crate::log;

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

pub fn init() {
    register_command!("help", help_command, "Prints help information", "help [command]");
}

fn help_command(args: Vec<String>) {
    let registry = COMMAND_REGISTRY.lock().unwrap();
    if args.is_empty() {
        for command in registry.values() {
            log(format!("{}: {} | {}", command.name, command.help, command.usage));
        }
    } else {
        for arg in args {
            if let Some(command) = registry.get(&arg[..]) {
                log(format!("{}: {} | {}", command.name, command.help, command.usage));
            } else {
                log(format!("Command not found: {}", arg));
            }
        }
    }
}