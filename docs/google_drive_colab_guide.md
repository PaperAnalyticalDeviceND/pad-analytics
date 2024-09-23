

# Google Drive Integration with Google Colab

This guide provides basic instructions on how to connect Google Colab to your Google Drive, as well as some useful commands to manage files and folders.

## 1. Connecting Google Colab to Google Drive

To connect your Google Drive to Google Colab, follow these steps:

1. Run the following command to mount your Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. After running this code, you will be prompted to authorize Colab to access your Google Drive. Follow the steps in the authorization window.

Once the Drive is mounted, your files and folders will be accessible under `/content/drive/My Drive/`.

---

## 2. Navigating Files and Folders

### Check Current Directory
To see the folder you're currently working in, use the following command:
```bash
!pwd
```

### List Files and Folders
To list the contents of your current directory, use:
```bash
!ls
```
To list the contents of a specific directory (e.g., `My Drive`), you can run:
```bash
!ls /content/drive/My\ Drive/
```

### Change Directory (Using %cd)
To change to a different directory, for example, to `My Drive`, use the `%cd` magic command:
```python
%cd /content/drive/My\ Drive/
```
The `%cd` command changes the current working directory just like in a terminal. Ensure that spaces are escaped with `\`.

---

## 3. Managing Files and Folders

Before performing any file management tasks, let's change to the directory where we want to work. For example, to navigate to the `My Drive` folder, use:

```python
%cd /content/drive/My\ Drive/
```

Once you're in the desired folder, you can use the following commands to manage your files and folders.

### Create a Folder
To create a new folder in the current working directory, use:

```bash
!mkdir new_folder_name
```

This will create a folder named `new_folder_name` inside your current directory (`My Drive` in this case).

### List Files and Folders
To view the contents of your current folder, run the following command:

```bash
!ls
```

This will list all files and folders in the current directory.

### Remove a Folder or File
To delete a folder (and all its contents) or a file, use the following command:

```bash
!rm -r folder_or_file_name
```

This will remove the folder or file located in your current directory.

---

## 4. Additional Commands

- **Check Disk Space:**
   To check the available disk space on your environment, use:
   ```bash
   !df -h
   ```

- **Check File Details:**
   To see more details (like file sizes or modification times), use:
   ```bash
   !ls -lh
   ```

---

By navigating to the correct folder first using `%cd`, you can manage files and folders relative to your current working location.
