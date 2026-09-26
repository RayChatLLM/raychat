# RayChat

## Start chatting

1. Install Python 3.10 or newer if it is not already installed.
2. Extract the **entire ZIP** to a folder of your choice.
3. Open a terminal in that folder and run:

   ```sh
   python raychat
   ```

These instructions are the same on Windows, macOS, and Linux. The `python`
command must refer to Python 3.10 or newer. There is no RayChat installer and
no package installation is needed for ordinary use.

The first-run window asks for your API token, model ID, and API base URL.
Paste each value and choose **Save and continue**. RayChat checks the provider's
models endpoint before saving. If it fails, the window explains the error and
keeps your entries for correction. Chat requires an internet connection and a
working OpenAI-compatible provider.

Settings are saved privately in `~/.raychat/environment/.env`. Run
`python raychat` again to start chatting with the saved settings. You do not
need to export environment variables or edit the installation.

## Choose a project

By default, RayChat uses a `workspace` folder in the terminal's current directory.
To work on another folder, including when the download folder is read-only:

```sh
python raychat --workspace "path/to/your/project"
```

Use **Enter** to send a message and `/quit` to exit. Writes and commands request
approval. `/models` lists available provider models. Run `python raychat --help`
for all options. Existing `--config`, `--env-file`, and `--portable` options work.

## What's in the download?

Keep `raychat` and `_raychat` together. `_raychat` contains the application,
licenses, pinned plugin profile, and support code required by recovery and
advanced core validation. `plugins` contains matching distributable plugin
archives and their catalog/profile. The standard plugins load automatically.
Advanced source-validation tools retain their existing development prerequisites;
they are not required for normal chat.

## Troubleshooting

- **Python is not found:** make your Python installation available as `python`
  in the terminal. Check `python --version`.
- **Missing application files:** extract the entire archive; do not launch from
  inside a ZIP viewer or move the `raychat` file by itself.
- **Setup rejects the token or URL:** check the provider's token permissions and
  API base URL, your connection, and whether it supports `GET /models`.
- **A different provider is used:** nonblank `RAYCHAT_*` variables already set in
  your shell take precedence over saved settings.
- **A folder is read-only:** choose a writable project with `--workspace`.
  `--env-file "path/to/settings.env"` selects another writable settings location.
