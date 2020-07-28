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
// This is a middleware general file storage. Images for inference will be copied out of here.
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

// SERVE STATIC FILES from ./public and ./inferenceContent
app.use(express.static('public'))
app.use(express.static('inferenceContent'))

// ROUTES
app.get('/', (req, res) => {
    res.render("index");
})

app.post('/form', upload.single('file'), async (req, res) => {
    const image = req.file
    //Set path to directory where images will be stored for and used for Protectron2
    const inputContentPath = './inferenceContent/input'

    console.log(image)
    if (fs.existsSync(inputContentPath)) {
        //If directory is already present:
        shell.exec(`cp ${image.path} ${inputContentPath} && python inference.py`, function(code, stdout, stderr) {
            if (stderr) {
                console.log('Program stderr:', stderr);
            } else {
                console.log('Success!')
                res.json({
                    //Output filename is the same. The directory is different, so:
                    image: `http://localhost:3000/output/${image.filename}`
                })
            }
          });
    } else {
        //If directory is not present:
        shell.exec(`mkdir ${inputContentPath} || cp ${image.path} ${inputContentPath}  && python inference.py`, function(code, stdout, stderr) {
            if (stderr) {
                console.log('Program stderr:', stderr);
            } else {
                console.log('Success!')
                res.json({
                    //Output filename is the same. The directory is different, so:
                    image: `http://localhost:3000/output/${image.filename}`
                })
            }
          });
    }


    //console.log('stdout:', stdout);
    //console.log('stderr:', stderr);
})

// SERVE APPLICATION
app.listen(3000, () => {
    console.log('App is working on port 3000')
})