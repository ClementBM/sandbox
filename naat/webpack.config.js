const path = require("path");

module.exports = {
    mode: "production",
    entry: {
        main: path.resolve("./nocodb-client.js"),
    },
    output: {
        path: path.resolve(__dirname, "dist"),
        filename: "Nocodb_sdk.js",
        library: {
            name: "Nocodb_sdk",
            type: "var",
        }
    }
}