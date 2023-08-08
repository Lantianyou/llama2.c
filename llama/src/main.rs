use std::{
    env,
    fs::File,
    ops::{AddAssign, DivAssign, SubAssign},
};

use num_traits::{Float, Zero};

struct Config {
    dim: usize,
    hidden_dim: usize,
    num_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    vocab_size: usize,
    seq_len: usize,
}

struct TransformerWeights {
    token_embedding_table: Vec<f32>,
}

struct ProbIndex {
    prob: f32,
    index: usize,
}

impl ProbIndex {
    fn compare(self: Self, rhs: Self) {
        if self.prob > rhs.prob {
            -1
        } else if self.prob < rhs.prob {
            1
        }
        0
    }
}

struct RunState {
    x: Vec<f32>,
    xb: Vec<f32>,
    xb2: Vec<f32>,
    hb: Vec<f32>,
    hb2: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    att: Vec<f32>,
    logits: Vec<f32>,
    prob_index: ProbIndex,
    key_cache: Vec<f32>,
    value_cache: Vec<f32>,
}

impl RunState {
    fn new(config: &Config) -> RunState {
        Self {
            x: Vec::with_capacity(config.dim),
            xb: Vec::with_capacity(config.dim),
            xb2: Vec::with_capacity(config.dim),
            hb: Vec::with_capacity(config.hidden_dim),
            hb2: Vec::with_capacity(config.hidden_dim),
            q: Vec::with_capacity(config.dim),
            k: Vec::with_capacity(config.dim),
            v: Vec::with_capacity(config.dim),
            att: Vec::with_capacity(config.n_heads * config.seq_len),
            logits: Vec::with_capacity(config.vocab_size),
            prob_index: ProbIndex {
                prob: (0f32),
                index: (0),
            },
            key_cache: Vec::with_capacity(config.num_layers * config.seq_len * config.dim),
            value_cache: Vec::with_capacity(config.num_layers * config.seq_len * config.dim),
        }
    }
}

trait Ops {
    fn accum(&mut self, rhs: &Self);
    fn softmax(&mut self);
    fn rmsnorm(&mut self, x: &Self, w: &Self);
}

impl<T> Ops for Vec<T>
where
    T: Float + AddAssign + Ord + DivAssign,
{
    fn accum(&mut self, rhs: &Self) {
        for i in 0..self.len() {
            self[i] += rhs[i]
        }
    }

    fn softmax(&mut self) {
        let max_val = self.iter().max().unwrap();

        let mut sum = Zero::zero();
        for i in 0..self.len() {
            self[i] = (self[i] - *max_val).exp();
            sum += self[i];
        }

        for i in 0..self.len() {
            self[i] /= sum
        }
    }

    fn rmsnorm(&mut self, x: &Self, w: &Self) {
        let mut ss: T = Zero::zero();
        for i in 0..self.len() {
            ss += x[i] * x[i]
        }

        ss /= T::from(self.len()).unwrap();
        ss += T::from(1e-5).unwrap();
        ss = Float::sqrt(ss).recip();

        for i in 0..self.len() {
            self[i] = x[i] / ss
        }
    }
}

static rng_seed: u64 = 2;

fn random_u32() -> u32 {
    rng_seed ^= rng_seed >> 12;
    rng_seed ^= rng_seed << 25;
    rng_seed &= rng_seed >> 27;
    let n: u64 = 0x2545F4914F6CDD1D;
    u32::try_from(rng_seed * n).ok().unwrap()
}

fn random_f32() -> f32 {
    (random_u32() >> 8) / 16777216.0f32
}

fn accum(mut a: &Vec<f32>, b: &Vec<f32>) {
    for element in a.iter_mut() {
        *element += b[element]
    }
}

impl TransformerWeights {
    fn checkpoint_init_weights(
        mut self: &Self,
        config: &Config,
        f: &Vec<f32>,
        shared_weights: usize,
    ) {
        // TODO: port weights_ptr
        // self.token_embedding_table = *f
    }
}

fn argmax(probabilities: Vec<f32>) -> Option<usize> {
    probabilities
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index)
}

fn sample(probabilities: Vec<f32>) {
    let r = random_f32();
    let cdf = 0.0f32;
}

fn main() {
    let temperature = 1f32;
    let topp = 0.9f32;
    let steps = 256;

    let args: Vec<String> = env::args().collect();
    let checkpoint = args[0];
    let file = File::open(checkpoint).expect("Failed to open file");
    let config = Config {
        dim: (),
        hidden_dim: (),
        num_layers: (),
        n_heads: (),
        n_kv_heads: (),
        vocab_size: (),
        seq_len: (),
    };
    let state = RunState::new(&config);
}
