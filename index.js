const express = require('express')
const multer = require('multer')
const mime = require('mime-types')
const shortid = require('shortid')
const { exec } = require('child_process')
const fs = require('fs')
var shell = require('shelljs');


// CREATE APPLICATION
const app = express()
app.set("view engine","ejs")

// FILE STORAGE
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, 'public')
    },
    filename: function (req, file, cb) {
        let id = shortid.generate()
        let ext = mime.extension(file.mimetype)
        cb(null, `${id}.${ext}`)
    }
})
const upload = multer({ storage })

// SERVE STATIC FILES
app.use(express.static('public'))

// ROUTES
app.get('/', (req, res) => {
    res.render("index");
})

app.post('/form', upload.single('file'), async (req, res) => {
    const image = req.file
    //Set path to directory where images will be stored for and used for Protectron2
    const path = 'importedImages'
    console.log(image)
    if (fs.existsSync(path)) {
        var { stdout, stderr } = await shell.exec(`cp ${image.path} inferenceContent/input && python inference.py`, function(code, stdout, stderr) {
            console.log('Exit code:', code);
            console.log('Program output:', stdout);
            console.log('Program stderr:', stderr);
          });
    } else {
        var { stdout, stderr } = await shell.exec(`mkdir inferenceContent/input || cp ${image.path} inferenceContent/input && python inference.py`, function(code, stdout, stderr) {
            console.log('Exit code:', code);
            console.log('Program output:', stdout);
            console.log('Program stderr:', stderr);
          });
    }


    //console.log('stdout:', stdout);
    //console.log('stderr:', stderr);

    res.json({
        image: `http://localhost:3000/${image.filename}`
    })
})

// SERVE APPLICATION
app.listen(3000, () => {
    console.log('App is working on port 3000')
})