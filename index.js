const express = require('express')
const multer = require('multer')
const mime = require('mime-types')
const shortid = require('shortid')
const { exec } = require('child_process')
const fs = require('fs')
const path = require('path')
var shell = require('shelljs');


// CREATE APPLICATION
const app = express()
app.set("view engine","ejs")

// FILE STORAGE
// This is a middleware general file storage. Images for inference will be copied out of here.
const storage = multer.diskStorage({
    //Handle default uploaded image destination.
    destination: function (req, file, cb) {
        cb(null, 'public')
    },
    //Handle uploaded image names by assigning a short id and extension.
    //todo: make items sortable.
    filename: function (req, file, cb) {
        let id = Date.now()+shortid.generate()
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

    //joining path of directory 

    const directoryPath = './inferenceContent/output/'
    const directoryShortPath = '/output/'

    //Passing directoryPath and callback function
    fs.readdir(directoryPath, function (err, files) {

        //handling error
        if (err) {
            console.log('Unable to find directory: ' + err + '. Output directory must exist in inferenceContent directory.');
        } 
        
        //set up array to store links to images, to show on main screen.
        var imageUrls = []
        var outputData = ''
        //push all files as dictionaries into list to easily parse them into .ejs template. Using forEach. :
        if (files) {
            var outputDataFilePath = 'inferenceContent/outputData/outputData.json'
            if(!fs.existsSync(outputDataFilePath)) {
                console.log("No inference output to display yet.");
              }
            else {
                fs.readFile(outputDataFilePath, (err, data) => {
                    if (err) throw err
                    outputData = JSON.parse(data)
                    console.log(outputData)
                });
              } 
            files.forEach(function (file) {
                //unshift newer images to appear on top, to avoid having to sort on client.
                imageUrls.unshift({
                    key: 'image',
                    value: directoryShortPath+file
                })
            })

        //if there are no files in directory:
        } else {
            console.log('No images to fetch from output directory. Upload an image for inference.')
        }

        //combine both data objects: imageUrls and outputData - into one, to be able to render index.ejs with them.
        
        //output list of images present at home directory load. Pass object as itself to index.ejs.
        res.render('index', {
            imageUrls,
            outputData
        })

    })
    
})

app.post('/form', upload.single('file'), async (req, res) => {
    const image = req.file
    //Set path to directory where images will be stored for and used for Protectron2 operations.
    const inputContentPath = './inferenceContent/input/'
    const imageNamePath = inputContentPath+image.filename
    console.log(imageNamePath)
    if (fs.existsSync(inputContentPath)) {
        //If directory is already present:
        shell.exec(`cp ${image.path} ${inputContentPath} && python inferenceSingle.py --image ${imageNamePath}`, function(code, stdout, stderr) {
            if (stderr) {
                console.log('Program stderr:', stderr)
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
        shell.exec(`mkdir ${inputContentPath} && cp ${image.path} ${inputContentPath}  && python inferenceSingle.py --image ${imageNamePath}`, function(code, stdout, stderr) {
            if (stderr) {
                console.log('Program stderr:', stderr)
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