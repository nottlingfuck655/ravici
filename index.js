const express = require('express');
const { HfInference } = require('@huggingface/inference');
const app = express();
const port = 3000;

app.use(express.json());
app.use(express.static('public'));

const hf = new HfInference(process.env['HF_API_KEY']);

app.post('/api/generate', async (req, res) => {
  try {
    const { prompt, guidanceScale } = req.body;

    const parameters = { 
      num_inference_steps: 25,
      guidance_scale: guidanceScale,
      negative_prompt: "low quality, blurry"
    };

    const imageBlob = await hf.textToImage({
      model: "stabilityai/stable-diffusion-xl-base-1.0",
      inputs: prompt,
      parameters: parameters
    });

    const buffer = await imageBlob.arrayBuffer();
    const base64 = Buffer.from(buffer).toString('base64');
    const imageUrl = `data:image/jpeg;base64,${base64}`;

    res.json({ images: [imageUrl] });
  } catch (error) {
    console.error('Error:', error);
    res.status(500).json({ 
      error: error.message,
      fullError: error
    });
  }
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});
