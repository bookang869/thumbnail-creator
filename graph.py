from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, interrupt, Command
from typing import TypedDict
from openai import OpenAI
import subprocess
import textwrap
from langchain.chat_models import init_chat_model
from typing_extensions import Annotated
import operator
import base64

# import .env variables
load_dotenv()

# initialize the LLM
llm = init_chat_model("openai:gpt-4o-mini")

class State(TypedDict):
  # the video input file
  video_file: str
  # the audio input file
  audio_file: str
  # the transcription of the video
  transcription: str
  # the summary of the chunks of transcription
  summaries: Annotated[list[str], operator.add]
  # the final summary of the transcription
  final_summary: str
  # thumbnail file names
  thumbnail_sketches: Annotated[list[str], operator.add]
  # thumbnail prompts
  thumbnail_prompts: Annotated[list[str], operator.add]
  # feedback from the user
  user_feedback: str
  # prompt used to generate the thumbnail user approved
  chosen_prompt: str

# extract the audio of mp4 file using ffmpeg
def extract_audio(state: State):

  # converts mp4 video into mp3 audio file
  output_file = state["video_file"].replace("mp4", "mp3")

  # ffmpeg command to extract audio from video
  # e.g. $ ffmpeg -i input.mp4 output.avi
  command = [
    "ffmpeg",
    "-i",
    state["video_file"],
    # filter to the audio
    "-filter:a",
    # speed up video (doesn't change the quality of transcription but cheaper since shorter video)
    "atempo=2.0",
    # answers yes to all prompts (do you want to overwrite the file if it already exists)
    "-y",
    output_file
  ]

  # run the command in the terminal
  subprocess.run(command)

  # update the 'audio_file' state
  return {
    "audio_file": output_file
  }

# transcribe the audio file using whisper
def transcribe_audio(state: State):

  client = OpenAI()
  
  # open the audio file
  # state["audio_file"] is the path to the audio file
  # file requires a file-like object
  # 'rb' - open the file for reading in binary mode
  with open(state["audio_file"], "rb") as audio:
    # create the transcription
    transcription = client.audio.transcriptions.create(
      # required - 'file', 'model'
      # optional - 'response_format', 'language', etc...
      file=audio,
      model="whisper-1",
      response_format="text",
      # The language of the input audio. 
      # Supplying the input language in ISO-639-1 (e.g. en) format will improve accuracy and latency.
      language="en",
      # helps the model to understand the context of the video
      prompt="Tottenham, Arsenal, North London Derby"
    )

  # update the 'transcription' state
  return {
    "transcription": transcription
  }

# dispatch the transcription into chunks to 'summarize_chunk' node
def dispatch_summarizers(state: State):
  transcription = state["transcription"]
  chunks = []

  # creates a list of chunks with appropriate id
  # textwrap - split the transcript into chunks of defined length
  for i, text in enumerate(textwrap.wrap(transcription, 500)):
    chunks.append({"id": i + 1, "text": text})

  # send the chunks to the 'summarize_chunk' node in parallel
  return [Send("summarize_chunk", chunk) for chunk in chunks]

# summarize each chunk of transcription
def summarize_chunk(chunk):
  chunk_id = chunk["id"]
  text = chunk["text"]

  # ask the LLM to summarize the text
  response = llm.invoke(
    f"""
    Please summarize the following text.

    Text: {text}
    """
  )
  
  # format the summary
  summary = f"[Chunk {chunk_id}] {response.content}"

  # update the 'summaries' state
  return {
    "summaries": [summary]
  }

# create the final summary of the transcription
def mega_summary(state: State):

  # combine all the summaries into one string
  all_summaries = "\n".join(state["summaries"])

  prompt = f"""
    You are given multiple summaries of different chunks from a video transcription.

    Please create a comprehensive final summary that combines all the key points.

    Individual summaries: {all_summaries} 
  """

  response = llm.invoke(prompt)

  return {
    "final_summary": response.content
  }

# dispatch the final summary to the 'dispatch_sketchers' node
def dispatch_artists(state: State):
  # send the final summary to the 'generate_thumbnails' node in parallel
  # n - number of thumbnails to generate
  n = 5
  return [Send("generate_thumbnails", {"id": i + 1, "summary": state["final_summary"]}) for i in range(n)]

# create the sketches of the final summary
def generate_thumbnails(args):
  concept_id = args["id"]
  summary = args["summary"]

  prompt = f"""
    Based on this video summary, create a detailed visual prompt for a Youtube thumbnail.

    Create a detailed prompt for generating a thumbnail image that would attract viewers. Include:
      - Main visual elements
      - Color scheme
      - Text overlay suggestions
      - Overall composition

    Summary: {summary}
  """

  response = llm.invoke(prompt)
  thumbnail_prompt = response.content

  client = OpenAI()

  # generate the thumbnail
  result = client.images.generate(
    model="gpt-image-1",
    prompt=thumbnail_prompt,
    quality="low",
    moderation="low",
    size="auto"
  )

  # decode the image bytes
  image_bytes = base64.b64decode(result.data[0].b64_json)

  file_name = f"thumbnail_{concept_id}.jpg"

  # save the image to the local directory
  with open(file_name, "wb") as f:
    f.write(image_bytes)

  # update the 'thumbnail_sketches' and 'thumbnail_prompts' states
  return {
    "thumbnail_sketches": [file_name],
    "thumbnail_prompts": [thumbnail_prompt]
  }

# get feedback from the user
def human_feedback(state: State):
  answer = interrupt({
    "chosen_thumbanil": "Which thumbnail do you like the most? Answer in its ID number.",
    "feedback": "Provide any feedback or changes you'd like for the final thumbnail."
  })

  chosen_prompt = answer["chosen_thumbnail"]
  user_feedback = answer["feedback"]

  return {
    "chosen_prompt": state["thumbnail_prompts"][int(chosen_prompt) - 1],
    "user_feedback": user_feedback,
  }

# generate a hd thumbnail given user's chosen prompt and feedback
def generate_hd_thumbnail(state: State):
  chosen_prompt = state["chosen_prompt"]
  user_feedback = state["user_feedback"]

  prompt = f"""
    You are a professional YouTube thumbnail designer. Take this original thumbnail prompt and
    create an enhanced version that incorporates the user's feedback.

    ORIGINAL PROMPT: {chosen_prompt}

    USER FEEDBACK TO INCORPORATE: {user_feedback}

    Create an enhanced prompt that:
      1. Maintains the core concept from the original prompt.
      2. Specifically addresses and implements the user's feedback requests.
      3. Adds professional YouTube thumbnail specifications:
        - High contrast and bold visual elements
        - Clear focal points that draw the eye
        - Professional lighting and composition
        - Optimal text placement and readability with generous padding from eddges
        - Colors that pop and grab attention
        - Elements that work well at small thumbnail sizes
        - IMPORTANT: Always ensure adequate white space/padding between any text and the image borders
  """

  response = llm.invoke(prompt)

  final_thumbnail_prompt = response.content

  client = OpenAI()

  # generate the thumbnail
  result = client.images.generate(
    model="gpt-image-1",
    prompt=final_thumbnail_prompt,
    quality="high",
    moderation="low",
    size="auto"
  )

  # decode the image bytes
  image_bytes = base64.b64decode(result.data[0].b64_json)

  # save the image to the local directory
  with open("thumbnail_final.jpg", "wb") as f:
    f.write(image_bytes)

  
# create the state graph
graph_builder = StateGraph(State)

# create the nodes using the predefined functions
graph_builder.add_node("extract_audio", extract_audio)
graph_builder.add_node("transcribe_audio", transcribe_audio)
graph_builder.add_node("summarize_chunk", summarize_chunk)
graph_builder.add_node("mega_summary", mega_summary)
graph_builder.add_node("generate_thumbnails", generate_thumbnails)
graph_builder.add_node("human_feedback", human_feedback)
graph_builder.add_node("generate_hd_thumbnail", generate_hd_thumbnail)

# create the edges between the nodes
graph_builder.add_edge(START, "extract_audio")
graph_builder.add_edge("extract_audio", "transcribe_audio")
graph_builder.add_conditional_edges("transcribe_audio", dispatch_summarizers, ["summarize_chunk"])
graph_builder.add_edge("summarize_chunk", "mega_summary")
graph_builder.add_conditional_edges("mega_summary", dispatch_artists, ["generate_thumbnails"])
graph_builder.add_edge("generate_thumbnails", "human_feedback")
graph_builder.add_edge("human_feedback", "generate_hd_thumbnail")
graph_builder.add_edge("generate_hd_thumbnail", END)


# compile the graph
graph = graph_builder.compile(name="thumbnail-creator")

