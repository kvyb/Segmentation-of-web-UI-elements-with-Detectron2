const express = require('express')
const multer = require('multer')
const mime = require('mime-types')
const shortid = require('shortid')
const { exec } = require('child_process')
const fs = require('fs')
const path = require('path')
var shell = require('shelljs')

// CREATE APPLICATION
const app = express()
app.set("view engine", "ejs")

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
        let id = Date.now() + shortid.generate()
        let ext = mime.extension(file.mimetype)
        cb(null, `${id}.${ext}`)
    }
})
const upload = multer({ storage })

// SERVE STATIC FILES from ./public and ./inferenceContent
app.use(express.static('public'))
app.use(express.static('inferenceContent'))

//Function for async read JSON
function readFileAsync(path) {
    return new Promise((resolve, reject) => {
        fs.readFile(path, (err, data) => {
            if (err) reject(err)
            resolve(JSON.parse(data))
        })
    })
}

// ROUTES
app.get('/', (req, res) => {

    //joining path of directory 

    const directoryPath = './inferenceContent/output/'
    const directoryShortPath = '/output/'

    //Passing directoryPath and callback function
    fs.readdir(directoryPath, async function (err, files) {

        //handling error
        if (err) {
            console.log('Unable to find directory: ' + err + '. Output directory must exist in inferenceContent directory.')
        }

        //set up array to store links to images, to show on main screen.
        var imageData = []
        //outputData will store output from inference for each file and all classes. This will allow to make counts.
        var outputData = ''
        //results contains all encountered classes in total for all inferences and shows statistics. See below.
        var results = []
        //push all files as dictionaries into list to easily parse them into .ejs template. Using forEach. :
        if (files.length) {
            var outputDataFilePath = 'inferenceContent/outputData/outputData.json'
            if (!fs.existsSync(outputDataFilePath)) {
                console.log("No inference output to display yet.")
            } else {
                try {
                    outputData = await readFileAsync(outputDataFilePath)
                    // Map objects
                    Object.keys(outputData.imageData).forEach(key => {
                        let items = outputData.imageData[key]
                        items.forEach(item => {
                            //split because Detectron2 lables contain predicted classname and confidence as string
                            let itemClassName = item.split(' ')[0]
                            let itemClassValue = item.split(' ')[1]
                            if (outputData.AllClasses.includes(itemClassName)) {
                                if (itemClassName in results) {
                                    //increment counter for existing class initialised below
                                    results[itemClassName].count++
                                    results[itemClassName].values.push(itemClassValue)
                                } else {
                                    //initialise if there is a class which has been predicted
                                    let value = { count: 1, values: [itemClassValue], average: 0 }
                                    results[itemClassName] = value
                                }
                            }
                        })
                    })

                    // Count average percision percentage for each found class
                    Object.keys(results).forEach(key => {
                        let item = results[key]
                        if (item.count > 1) {
                            let total = item.values.reduce((acc, value) => {
                                return acc + parseInt(value.split('%')[0])
                            }, 0)
                            let average = total / item.values.length
                            results[key].average = average
                        } else {
                            results[key].average = parseInt(item.values[0].split('%')[0])
                        }
                    })

                } catch (error) {
                    throw error
                }
            }
            files.forEach(function (file) {
                //imageData is composed here to transfer data to our index.ejs template.
                //Data from here will be used to render our html file.
                //unshift newer images to appear on top, to avoid having to sort on client.
                //add image statistical data from inference to each image, parsed above, from outputData.
                imageData.unshift({
                    key: 'image',
                    name: file,
                    classesFound: outputData.imageData[file],
                    url: directoryShortPath + file

                })
            })

            //if there are no files in directory:
        } else {
            console.log('No images to fetch from output directory. Upload an image for inference.')
        }

        console.log('results:', results)
        console.log('outputdata:', outputData)
        console.log('imageData:', imageData)

        //output list of images present at home directory load. Pass object as itself to index.ejs.
        res.render('index', {
            //results for overall averages and class information. 
            //Takes into account all inferenced images since project has been wiped.
            results,
            imageData
        })

    })

})


app.post('/form', upload.single('file'), async (req, res) => {
    const image = req.file
    //Set path to directory where images will be stored for and used for Protectron2 operations.
    const inputContentPath = './inferenceContent/input/'
    const imageNamePath = inputContentPath + image.filename
    console.log(imageNamePath)
    if (fs.existsSync(inputContentPath)) {
        //If directory is already present:
        shell.exec(`cp ${image.path} ${inputContentPath} && python inferenceSingle.py --image ${imageNamePath}`, function (code, stdout, stderr) {
            if (stderr) {
                console.log('Program stderr:', stderr)
                return res.json({
                    success: false
                })
            } else {
                console.log('Success!')
                return res.json({
                    //Output filename is the same. The directory is different, so:
                    image: `http://localhost:3000/output/${image.filename}`
                })
            }
        })
    } else {
        //If directory is not present:
        shell.exec(`mkdir ${inputContentPath} && cp ${image.path} ${inputContentPath}  && python inferenceSingle.py --image ${imageNamePath}`, function (code, stdout, stderr) {
            if (stderr) {
                console.log('Program stderr:', stderr)
                return res.json({
                    success: false
                })
            } else {
                console.log('Success!')
                return res.json({
                    //Output filename is the same. The directory is different, so:
                    image: `http://localhost:3000/output/${image.filename}`
                })
            }
        })
    }


    //console.log('stdout:', stdout)
    //console.log('stderr:', stderr)
})

// SERVE APPLICATION
app.listen(3000, () => {
    console.log('App is working on port 3000')
})