import { Api } from 'nocodb-sdk'
// const nocodb_sdk = require('nocodb-sdk');

const api = new Api({
    baseURL: 'https://noco-db-production-ab17.up.railway.app',
    headers: {
      'xc-token': 'BHYjlOqT9XeLmgpkb1yMnDKfKM_TSNg34-P-OJ4Q'
    }
})

const projectId = "p_bpmgum5cfqxaos"

console.log(Object.getOwnPropertyNames(api.public.sharedBaseGet))

import * as fs from 'fs'

// console.log(await api.dbTableRow.list(projectName:"NAAT", orgs="", tableName="LegalCase"))

// https://noco-db-production-ab17.up.railway.app/dashboard/#/nc/view/2a4ae38c-370a-4d4c-9f7b-c935f89a9411

try {
    const data = await api.public.dataList("2a4ae38c-370a-4d4c-9f7b-c935f89a9411")
    console.log(data)
} catch (error) {
    console.log(error)
    fs.writeFile('output.json', JSON.stringify(error), (err) => {
        if (err) throw err;
    });
}

// console.log(await api.dbTableRow.list())

// export { nocodb_sdk }

// Example: Calling API - /api/v1/db/meta/projects/{projectId}/tables
// await api.dbTable.create(params)