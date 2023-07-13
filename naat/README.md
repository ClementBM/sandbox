# Introduction

## Nocodb

### Docker
```shell
docker build -t nocodbtest .

docker run -d --name nocodb -p 8080:8080 nocodbtest:latest
docker run -it --name nocodb -v "$(pwd)"/nocodb/:/usr/app/data/ -p 8080:8080 nocodbtest:latest

docker run -it --name nocodbtest -p 8080:8080 nocodbtest:latest
docker run -it --name nocodbtest -p 8080:8080 nocodb_climate_litigation:latest

docker stop nocodb && docker rm nocodb
docker image remove nocodbtest:latest
```


```shell
# Create an Image From a Container
docker commit <id_container>3
# Tag the Image
docker tag 9565323927bf nocodb_climate_litigation
```

### Setup Railway CLI
```shell
curl -fsSL https://railway.app/install.sh | sh
railway connect
railway link 1abe047c-0a00-4476-aff7-b5af6a273ab3
```

# Google Apps Script

[Mastering npm modules in google apps script (medium)](https://medium.com/geekculture/the-ultimate-guide-to-npm-modules-in-google-apps-script-a84545c3f57c)
[Write Google Apps Script Locally & Deploy with Clasp (medium)](https://medium.com/geekculture/how-to-write-google-apps-script-code-locally-in-vs-code-and-deploy-it-with-clasp-9a4273e2d018)


## Developer Resources
[nocodb rest apis](https://docs.nocodb.com/developer-resources/rest-apis/)
[nocodb sdk](https://docs.nocodb.com/developer-resources/sdk/)