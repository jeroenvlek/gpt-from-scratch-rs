use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    #[arg(short, long)]
    pub input_path: String,

    #[arg(long, default_value_t = 100)]
    pub num_epochs_simple: usize,

    #[arg(long, default_value_t = 5000)]
    pub num_epochs_complete: usize,

    #[arg(long, default_value_t = 5000)]
    pub max_new_tokens: usize,
}
