# Why?

I wanted to learn [Candle](https://github.com/huggingface/candle), so I decided to port Andrej Karpathy's [tutorial](https://youtu.be/kCc8FmEb1nY?si=0Glj-AuTc9fw8eZk). It gives a good feel for
the Candle API and it's meant for people who want to use Candle and see how certain things can be done, so 
do look into that _src_ directory!

Blogpost on [Perceptive Bits](https://www.perceptivebits.com/building-gpt-from-scratch-in-rust-and-candle/).

# How

This project was developed using Candle 0.3.2 and Rust 1.75.0 

You can run it with

`run --package gpt-from-scratch-rs --bin gpt-from-scratch-rs -- --input-path /home/jvlek/dev/datasets/tinyshakespeare.txt`

Where _input-path_ points to a text file you want to train on. See the file _src/args.rs_ for more arguments 
like number of epochs (note that you need to convert underscores to dashes).  
