const express = require('express')
const multer = require('multer')
const mime = require('mime-types')
const shortid = require('shortid')
const { exec } = require('child_process')

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

    const { stdout, stderr } = await exec('mkdir test');

    console.log('stdout:', stdout);
    console.log('stderr:', stderr);

    res.json({
        image: `http://localhost:3000/${image.filename}`
    })
})

// SERVE APPLICATION
app.listen(3000, () => {
    console.log('App is working on port 3000')
})
