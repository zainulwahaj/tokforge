use tokforge::{
    work::{Cancellation, CpuPool},
    EncodeOptions, Engine,
};

pub fn pool(workers: usize, queue: usize) -> CpuPool {
    let engine = Engine::from_bytes(
        include_bytes!("../fixtures/wordpiece.json"),
        EncodeOptions::default(),
    )
    .unwrap();
    CpuPool::new(engine, workers, queue, Cancellation::default()).unwrap()
}
