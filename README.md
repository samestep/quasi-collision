# SHA-256 quasi-collision

My answer to [this Cryptography Stack Exchange question](https://crypto.stackexchange.com/q/119199/48650) from December 2025.

Once you have [Rust](https://rust-lang.org/) and [Python](https://www.python.org/) installed (or you can just use the [Nix](https://nixos.org/) dev shell), run this command:

```sh
cargo run --release | tee check.py
```

Let it run however long you like, then stop it with `^C`. To validate the results, run this other command:

```sh
python3 check.py
```

For example, on my machine with an AMD Ryzen 9 9950X CPU:

```
$ tail -1 check.py
check(186, "003b668c", "004347a2")  # found after 4798.76804938s
```
