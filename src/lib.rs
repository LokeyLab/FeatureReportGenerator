pub mod lib_src;

#[allow(non_snake_case)]
pub fn hello() {
    println!("hello");
}

pub fn wrapper() {
    lib_src::tool::do_somthing()
}

pub use lib_src::tool::do_somthing;
