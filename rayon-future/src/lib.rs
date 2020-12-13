use std::pin::Pin;
use std::sync::{Arc, Mutex, mpsc};
use std::task::{Context, Poll, Waker};
use std::future::Future;




// ============================================================================
struct State<T> {
    recv: mpsc::Receiver<T>,
    waker: Option<Waker>,
}




// ============================================================================
pub struct RayonFuture<T> {
    state: Arc<Mutex<State<T>>>,
}




// ============================================================================
fn spawn_as_future_in_scope<'a, 'b, T, F>(scope: &'a rayon::Scope<'b>, f: F) -> RayonFuture<T>
    where T: 'b + Send,
          F: 'b + Send + FnOnce() -> T,
{
    let (send, recv) = mpsc::sync_channel(1);

    let state = State::<T>{
        recv: recv,
        waker: None,
    };
    let state = Arc::new(Mutex::new(state));
    let share = state.clone();

    scope.spawn(move |_| {
        send.send(f()).expect("send over channel");

        let mut share = share.lock().expect("lock on shared state");

        if let Some(waker) = share.waker.take() {
            waker.wake();
        };
    });

    RayonFuture{
        state: state,
    }
}




// ============================================================================
impl<T> Future for RayonFuture<T>
{
    type Output = T;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<T>
    {
        let mut state = self.state.lock().expect("lock on shared state");

        match state.recv.try_recv() {
            Ok(r) => Poll::Ready(r),
            Err(mpsc::TryRecvError::Empty) => {
                state.waker = Some(cx.waker().clone());
                Poll::Pending
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                panic!("unexpected rayon future hangup");
            }
        }
    }
}




// ============================================================================
pub trait FutureSpawn<'b, T> {
    type Future: Future;

    fn spawn_as_future<F: 'b + Send + FnOnce() -> T>(&self, f: F) -> Self::Future;
    fn run<F: 'b + Send + FnOnce() -> T>(&self, f: F) -> Self::Future;
}




// ============================================================================
impl<'b, T> FutureSpawn<'b, T> for rayon::Scope<'b> where T: 'b + Send
{
    type Future = RayonFuture<T>;

    fn spawn_as_future<F: 'b + Send + FnOnce() -> T>(&self, f: F) -> Self::Future {
        spawn_as_future_in_scope(self, f)
    }
    fn run<F: 'b + Send + FnOnce() -> T>(&self, f: F) -> Self::Future {
        spawn_as_future_in_scope(self, f)
    }
}




// ============================================================================
#[cfg(test)]
mod tests {
    use crate::FutureSpawn;
    use futures::executor::block_on;

    #[test]
    fn can_spawn_as_future_in_pool() {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().expect("pool");
        let data = String::from("12");

        pool.scope(|scope| {
            let fut1 = scope.spawn_as_future(|| &data);
            let fut2 = scope.spawn_as_future(|| &data);
            assert_eq!(block_on(fut1), "12");
            assert_eq!(block_on(fut2), "12");
        });
    }
}
