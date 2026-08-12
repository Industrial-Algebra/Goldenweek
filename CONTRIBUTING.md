# Contributing to Goldenweek

Thank you for your interest in contributing to Goldenweek!

## CLA

All contributors must sign the [Contributor License Agreement](https://github.com/Industrial-Algebra/.github/blob/main/CLA.md) before contributions can be merged.

## Development

```sh
git clone https://github.com/Industrial-Algebra/Goldenweek.git
cd Goldenweek
cargo test
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --check
```

(Backend tests, `--features metal` / `--features vulkan`, land with the render
backends in v0.2.)

## Git Flow

- Feature branches: `feature/<description>` off `develop`
- Hotfixes: `hotfix/<description>` off `main`
- PRs target `develop` for features, `main` for hotfixes
- Release PRs go from `develop` to `main`

## License

All contributions are licensed under [Apache-2.0](./LICENSE).
