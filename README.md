### Step 1: Start the python server

```sh

uvicorn main:app --host 0.0.0.0 --port 8000

```
## Step 2: Start Metro

First, you will need to run **Metro**, the JavaScript build tool for React Native.

To start the Metro dev server, run the following command from the root of your React Native project:

```sh
npx react-native start

```

## Step 3: Build and run your app

With Metro running, open a new terminal window/pane from the root of your React Native project, and use one of the following commands to build and run your Android app:


### Android

```sh
npx react-native run-android
```
