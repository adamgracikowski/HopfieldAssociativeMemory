//! Measurement of the heap used by a piece of work.

/// Source of the size of the heap, e.g. a counting global allocator of the
/// binary.
pub trait HeapProbe {
    /// Resets the peak to the current heap size and returns that size in
    /// bytes, or `None` if the heap is not measured.
    fn start(&self) -> Option<usize>;

    /// Largest heap size in bytes since the last [`start`](Self::start), or
    /// `None` if the heap is not measured.
    fn peak(&self) -> Option<usize>;

    /// Runs `work` and returns its result with the heap growth during it
    /// (peak minus start), or `None` if the heap is not measured.
    fn measure<T>(&self, work: impl FnOnce() -> T) -> (T, Option<usize>)
    where
        Self: Sized,
    {
        let start = self.start();
        let result = work();
        let growth = start
            .zip(self.peak())
            .map(|(start, peak)| peak.saturating_sub(start));
        (result, growth)
    }
}

/// A probe that measures nothing.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoProbe;

impl HeapProbe for NoProbe {
    fn start(&self) -> Option<usize> {
        None
    }

    fn peak(&self) -> Option<usize> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    /// A probe whose heap starts at 100 bytes and peaks at 150.
    struct Fixed {
        starts: Cell<usize>,
    }

    impl HeapProbe for Fixed {
        fn start(&self) -> Option<usize> {
            self.starts.set(self.starts.get() + 1);
            Some(100)
        }

        fn peak(&self) -> Option<usize> {
            Some(150)
        }
    }

    #[test]
    fn measures_the_growth_of_the_heap() {
        let probe = Fixed {
            starts: Cell::new(0),
        };
        assert_eq!(probe.measure(|| 7), (7, Some(50)));
        assert_eq!(probe.starts.get(), 1);
        assert_eq!(NoProbe.measure(|| 7), (7, None));
    }
}
