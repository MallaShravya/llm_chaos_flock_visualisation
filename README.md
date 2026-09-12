# LLM Chaos

We asked a computer to write a story. Then we asked it again.

Both times we asked the exact same thing. The only difference was **one extra space** at the end
of the question.

```
Describe a bird flying over a city...goes.
Describe a bird flying over a city...goes. 
                                          ^
                                   one space
```

You would expect nothing to change. But the computer wrote the same story only for a while,
and then the two stories went completely different ways.

| | Story A | Story B |
|---|---|---|
| Same for | the first 107 pieces | the first 107 pieces |
| Then | *"workers toiling away at their desks"* | *"workers pushing heavy boxes up and down stairs"* |
| Ending | the bird sleeps under the stars | a man talking in the distance |

(Computers write in small pieces called **tokens** — usually a word, or part of one.)

## The birds

Every piece the computer wrote becomes a bird.

Story A's pieces make one flock. Story B's pieces make another. The two flocks fly next to each
other on the screen. For a long time they move **exactly** together — same shape, same turns.
Then the stories split, and so do the flocks. They never fly together again.

Each new bird is born outside the flock. The three birds whose words mean the most similar thing
fly out to meet it and bring it in. So the shape of the flock is really the shape of the story.

## Try it

You need Python and Node. Run the first four from the main folder:

```bash
pip install -r requirements.txt
python pipeline/project_embeddings.py
python pipeline/simulate_semantic_flock_optimized_v2.py
python pipeline/convert_motion_to_binary_v2.py

cd viewer && npm install && npm run dev
```

Then open the link it prints. Step 3 takes a few minutes and makes a big file; step 4 shrinks it.

The two stories are already saved here (`run_a.json`, `run_b.json`), so you do **not** need a
fancy graphics card.

## Change the question

Want to see if a comma does it? Or a typo? Open `pipeline/modal_qwen.py`, change the two
questions near the top, and run the steps again. This part needs a free
[Modal](https://modal.com) account, because it borrows a big computer for about a minute.

Fun one to try: make both questions **identical**. The two stories should then stay the same
forever. If they don't, something in your setup is random when it shouldn't be.

## License

MIT — see [LICENSE](LICENSE).
