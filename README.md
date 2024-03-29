# U^2-net in Browser

This project attempts to run [U2Net](https://github.com/xuebinqin/U-2-Net) in a web browser. The project is based on the reduced-size U2Net model proposed in [silueta.me](https://github.com/xuebinqin/U-2-Net/issues/295), and is inferencing in the browser through the ONNX web runtime.

You can try online: [https://jerrychan7.github.io/U2netInBrowser/](https://jerrychan7.github.io/U2netInBrowser/).
**After the web page is opened, the model (*42MB in size*) will be loaded directly. Please open it when the network environment is suitable, such as WiFi.**

## Development

This project was bootstrapped with [Create React App](https://github.com/facebook/create-react-app) and [React App Rewired](https://github.com/timarney/react-app-rewired).

In the project directory, you can run:

### `npm start`

Runs the app in the development mode.\
Open [http://localhost:3000](http://localhost:3000) to view it in your browser.

The page will reload when you make changes.\
You may also see any lint errors in the console.

### `npm run build`

Builds the app for production to the `build` folder.\
It correctly bundles React in production mode and optimizes the build for the best performance.

The build is minified and the filenames include the hashes.\
Your app is ready to be deployed!

See the section about [deployment](https://facebook.github.io/create-react-app/docs/deployment) for more information.

### `npm run eject`

**Note: this is a one-way operation. Once you `eject`, you can't go back!**

If you aren't satisfied with the build tool and configuration choices, you can `eject` at any time. This command will remove the single build dependency from your project.

Instead, it will copy all the configuration files and the transitive dependencies (webpack, Babel, ESLint, etc) right into your project so you have full control over them. All of the commands except `eject` will still work, but they will point to the copied scripts so you can tweak them. At this point you're on your own.

You don't have to ever use `eject`. The curated feature set is suitable for small and middle deployments, and you shouldn't feel obligated to use this feature. However we understand that this tool wouldn't be useful if you couldn't customize it when you are ready for it.

### `npm run deploy`

Build the React application, put the result into the gh-pages branch, and push it to GitHub for GitHub Pages visiting.
