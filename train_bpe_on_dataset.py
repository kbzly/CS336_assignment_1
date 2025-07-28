from cs336_basics.tokenizer import train_bpe
import time
import cProfile
import pstats

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    address = "tests/fixtures/tinystories_sample_5M.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]

    time_start = time.time()
    train_bpe(address, vocab_size, special_tokens)

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.print_stats(30)

    time_end = time.time()
    print(f"Time taken: {time_end - time_start} seconds")