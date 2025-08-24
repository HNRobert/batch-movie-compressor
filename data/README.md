# Data Directory

This directory contains subdirectories for input and output movie files.

## Structure

```text
data/
├── in/                           # Input directory - put original movie files here
│   ├── Movie1.mkv
│   ├── Movie2.mp4
│   └── ...
└── out/                          # Output directory - compressed files go here
    ├── Movie1_AV1_4K_HDR10.mkv
    ├── Movie2_AV1_4K_HDR10.mkv
    ├── compression.log           # Compression log file
    └── ...
```

## Usage

1. Put your original movie files in the `data/in/` folder
2. Run the compression tool
3. Find compressed files in the `data/out/` folder

## Notes

- The `in/` and `out/` directories will be created automatically if they don't exist
- Original files are never modified or deleted
- Compression logs are saved as `compression.log` in the output directory
