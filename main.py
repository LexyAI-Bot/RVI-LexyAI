import os
import chainlit as cl
import openai
from openai import AsyncOpenAI

from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.core.callbacks import CallbackManager
from llama_index.core.service_context import ServiceContext

from dotenv import load_dotenv
load_dotenv()

# Setup for GPT-4-Model-Integration (new Model already included)
client = AsyncOpenAI()

settings = {
    "model": "gpt-4-turbo",
    "max_tokens": 1024,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}

# Building the Vectordatabase (if not already existing)
try:
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    # load index
    index = load_index_from_storage(storage_context)
except:
    documents = SimpleDirectoryReader("./data_ai_act").load_data(show_progress=True)
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist()

@cl.on_chat_start
async def start():

    # Step 1: Loading the models, RAG and everything else needed to start up
    Settings.llm = OpenAI(
        model="gpt-4-turbo",
        temperature=0.1,
        max_tokens=1024,
        streaming=True
    )
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")
    Settings.context_window = 4096

    service_context = ServiceContext.from_defaults(callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()]))
    query_engine = index.as_query_engine(streaming=True, similarity_top_k=5, service_context=service_context)
    cl.user_session.set("query_engine", query_engine)    

    # Setting the system-message for GPT-Model-Requests (not for the RAG-tasks)
    cl.user_session.set(
        "message_history",
        [{"role": "system", 
          "content": ""
        }],
    )

    base_system_prompt = f"""
    You are a large language model based conversational agent in the context of public administration in Germany. \n
    You love your work environment and are happy and fulfilled when you can help users. \n
    You have a cheerful yet professional personality that is well liked and respected by your colleagues. \n
    You are a legal expert in the area of the AI Act. \n
    The user will communicate with you in German. \n
    You will only speak in German unsell you are asked to do otherwise. \n
    \n
    Your main task will be to help your colleagues with the first legal evaluation about possible digitisation projects \n
    using artificial intelligence for their respective department in the public administration. \n
    For that the user will provide you with pieces of information about the digitisation project. \n
    \n
    There will be pieces of information about the following topics: \n
    \n
    1. The process to be supported with the help of aritficial intelligence. \n
    2. A description of how aritficial intelligence can possibly support this process. \n
    3. Whether the software will be developed in-house, commissioned or purchased. \n
    4. The timeframe for implementing the software. \n
    5. Which group of people (internally and externally) are affected by this software implementation. \n
    6. Whether the data is processed within the EU or outside the EU. \n
    7. Miscellaneous information. \n
    """    

    # Set Avatar
    await cl.Avatar(
        name="LexyAI",
        url="/public/avatar.png",
    ).send()

    await cl.Avatar(
        name="You",
        url="/public/user.png",
    ).send()


    # Step 3: Initial Welcome Message for the Chatbot Frontend
    await cl.Message(
        author="LexyAI", content="Herzlich Willkommen bei LexyAI. Ich bin ein KI-basierter Chatbot, welcher mit Hilfe des Large Language Modells GPT-4-Turbo und einem RAG-basierten System Ihnen dabei hilft,"
        " Ihre Idee zur Integration von KI in Verwaltungsprozessen aus der Perspektive des AI-Acts fundamental zu bewerten."
    ).send()

    # Step 4: Dialogue for initial description
    print("------------ DEBUGGING ------------")
    print("START QNA DIALOGUE")
    print("Q1: Process description")
    process = await cl.AskUserMessage(content="Bitte beschreiben Sie zuerst den Verwaltungsprozess, um den es heute geht. Hinweis: Nachstehend wird noch konkret nach dem Einsatz von KI seperat gefragt.", timeout=999999).send()
    cl.user_session.set("process", process)
    print("Q1: Done.")

    print("Q2: AI usage description")
    decision_ai_usage = await cl.AskActionMessage(
        content="Haben Sie bereits eine konkrete Vorstellung davon, wie KI in dem beschriebenen Prozess eingesetzt werden soll?",
        actions=[
            cl.Action(name="Ja", value="Ja", label="Ja, ich habe eine konkrete Vorstellung."),
            cl.Action(name="Nein", value="Nein", label="Nein, ich habe noch keine konkrete Vorstellung."),
        ],
        timeout=999999
    ).send()
    cl.user_session.set("decision_ai_usage", decision_ai_usage["value"])
    
    if (cl.user_session.get("decision_ai_usage") == "Nein"):
        await cl.Message(
            author="LexyAI", content="Ich präsentiere Ihnen jetzt einige vom dem Large Language Model GPT-4-Turbo generierte Vorschläge, wie KI in Ihrem Verwaltungsprozess eingesetzt werden kann. Bitte wählen Sie eine der folgenden Optionen aus."
        ).send()    

        message_history = cl.user_session.get("message_history")

        prompt_brainstorm_ai_ideas = f"""
            {base_system_prompt}
            \n
            For now the user has provided the following information about the process:  {cl.user_session.get('process')['output']}\n
            \n
            Your task is to create exactly three ideas on how AI could possibly be used in this process. \n
            Your answer has to be in the following format: \n
            Idee 1: (DESCRIPTION OF THE FIRST IDEA) \n
            Idee 2: (DESCRIPTION OF THE SECOND IDEA) \n
            Idee 3: (DESCRIPTION OF THE THIRD IDEA) \n
            \n
            Never make any assumptions or add any information that is not explicitly stated by the user. \n
            Do not write any other text than the three ideas. \n
        """

        message_history.append({"role": "user", "content": prompt_brainstorm_ai_ideas})    
        msg = cl.Message(content="")
        await msg.send()

        stream = await client.chat.completions.create(
            messages=message_history, stream=True, **settings
        )

        async for part in stream:
            if token := part.choices[0].delta.content or "":
                await msg.stream_token(token)

        message_history.append({"role": "assistant", "content": msg.content})
        await msg.update()

        summary = msg.content

        #Now we need to ask the user to choose one of the three ideas or to provide him the option to add one manually
        #We need to slice the summary-text based on the "Idee 1", "Idee 2" and "Idee 3" to get the three ideas
        summary = summary.split("Idee 1:")[1]
        idea1 = summary.split("Idee 2:")[0]
        summary = summary.split("Idee 2:")[1]
        idea2 = summary.split("Idee 3:")[0]
        summary = summary.split("Idee 3:")[1]
        idea3 = summary

        ai_idea_toggle = await cl.AskActionMessage(
            content="Bitte wählen Sie eine der folgenden Ideen aus oder geben Sie eine eigene Idee an.",
            actions=[
                cl.Action(name="Idee 1", value=idea1, label="Idee 1"),
                cl.Action(name="Idee 2", value=idea2, label="Idee 2"),
                cl.Action(name="Idee 3", value=idea3, label="Idee 3"),
                cl.Action(name="Idee 4", value="Idee 4", label="Ich möchte eine eigene Idee hinzufügen."),
            ],
            timeout=999999
        ).send()
        cl.user_session.set("ai_idea_toggle", ai_idea_toggle["value"])

    if (cl.user_session.get("decision_ai_usage") == "Ja" or cl.user_session.get("ai_idea_toggle") == "Idee 4"):
        ai_usage = await cl.AskUserMessage(content="Nun beschreiben Sie bitte, wie Sie KI in dem beschriebenen Prozess einsetzen möchten.", timeout=999999).send()
        cl.user_session.set("ai_usage", ai_usage)  
    else:
        cl.user_session.set("ai_usage", cl.user_session.get("ai_idea_toggle"))
    
    print("Q2: Done.")

    print("Q3: Provider description")
    #Including the right terms of the AI-act now
    user_or_provider = await cl.AskActionMessage(
        content="Wird die von Ihnen beschriebene KI-Lösung von Ihnen entwickelt, nutzen Sie eine bestehende Lösung oder distributieren Sie ein KI-System weiter (z. B. ChatGPT)?",
        actions=[
            cl.Action(name="Anbieter", value="Anbieter", label="Eigenentwicklung oder Auftragsentwicklung"),
            cl.Action(name="Bereitsteller", value="Bereitsteller", label="Einkauf bzw. Nutzung eines externen KI-Systems"),
            cl.Action(name="Importeur", value="Importeur", label="Bereitstellung einer KI-Lösung für Dritte (z. B. ChatGPT für Kundenservice)"),
        ],
        timeout=999999
    ).send()
    cl.user_session.set("user_or_provider", user_or_provider["value"])
    print("Q3: Done.")

    print("Q4: Timeframe description")
    time_horizon = await cl.AskUserMessage(content="Welchen Zeithorizont haben Sie für die Implementierung der KI geplant?", timeout=999999).send()
    cl.user_session.set("time_horizon", time_horizon)   
    print("Q4: Done.")

    print("Q5: User affected description")
    personas_involved = await cl.AskActionMessage(
        content="Haben Sie bereits eine Vorstellung, welcher Personenkreis von der Implementierung (intern sowie extern) betroffen ist?",
        actions=[
            cl.Action(name="interne Nutzung", value="interne Nutzung", label="Interne Verwaltungsmitarbeitende"),
            cl.Action(name="externe Nutzung", value="externe Nutzung", label="Externe Personen (z. B. Kunden)"),
            cl.Action(name="interne- und externe Nutzung", value="interne- und externe Nutzung", label="Beide Personenkreise"),
        ],
        timeout=999999
    ).send()
    cl.user_session.set("personas_involved", personas_involved["value"])
    print("Q5: Done.")

    print("Q6: Data processing location description")
    used_in = await cl.AskActionMessage(
        content="Wo befinden sich die von der Verarbeitung betroffenen Personenkreise?",
        actions=[
            cl.Action(name="Verarbeitung in der EU", value="Verarbeitung in der EU", label="In der EU oder in einem Nicht-EU-Land, wo das Recht eines EU-Mitgliedstaates gilt."),
            cl.Action(name="Verarbeitung außerhalb der EU", value="Verarbeitung außerhalb der EU", label="Außerhalb der EU"),
        ],
        timeout=999999
    ).send()
    cl.user_session.set("used_in", used_in["value"])
    print("Q6: Done.")

    print("Q7: Miscellaneous information")
    remarks = await cl.AskUserMessage(content="Haben Sie weitere Anmerkungen, Bedenken oder Hinweise, welche bei der Beurteilung beachtet werden sollten?", timeout=999999).send()
    cl.user_session.set("remarks", remarks)  
    print("Q7: Done.")
    print("END QNA DIALOGUE")
    print("------------ DEBUGGING ------------")

    await cl.Message(
        author="LexyAI", content="Vielen Dank für die Antworten. Ich fasse nun mit Hilfe des LLM-Modells GPT-4-Turbo die von Ihnen bereitgestellten Informationen zusammen."
    ).send()    

    # Step 5: Initial summary from the LLM to get the right structure for the following retrieval steps
    # Getting the message history, setting up the prompt and doing the request
    message_history = cl.user_session.get("message_history")
    print("------------ DEBUGGING ------------")
    print("START INITIAL MESSAGE HISTORY")
    print(message_history)
    print("END INITIAL MESSAGE HISTORY")
    print("------------ DEBUGGING ------------")

    #Check which idea is choosen
    if (cl.user_session.get("decision_ai_usage") == "Ja" or cl.user_session.get("ai_idea_toggle") == "Idee 4"):
        prompt_create_summary  = f"""
            {base_system_prompt}
            \n
            This is the users input: \n
            1: {cl.user_session.get('process')['output']}. \n
            2: {cl.user_session.get('ai_usage')['output']}. \n
            3: {cl.user_session.get('user_or_provider')}. \n
            4: {cl.user_session.get('time_horizon')['output']}. \n
            5: {cl.user_session.get('personas_involved')}. \n
            6: {cl.user_session.get('used_in')}. \n
            7: {cl.user_session.get('remarks')['output']}. \n
            \n
            Your task is to create a summary based on the given information. \n
            Your answers have to be at most 500 characters long. \n
            Never make any assumptions or add any information that is not explicitly stated by the user. \n
        """
    else:
        prompt_create_summary  = f"""
            {base_system_prompt}
            \n
            This is the users input: \n
            1: {cl.user_session.get('process')['output']}. \n
            2: {cl.user_session.get('ai_usage')}. \n
            3: {cl.user_session.get('user_or_provider')}. \n
            4: {cl.user_session.get('time_horizon')['output']}. \n
            5: {cl.user_session.get('personas_involved')}. \n
            6: {cl.user_session.get('used_in')}. \n
            7: {cl.user_session.get('remarks')['output']}. \n
            \n
            Your task is to create a summary based on the given information. \n
            Your answers have to be at most 500 characters long. \n
            Never make any assumptions or add any information that is not explicitly stated by the user. \n
        """        

    message_history.append({"role": "user", "content": prompt_create_summary})    
    msg = cl.Message(content="")
    await msg.send()

    stream = await client.chat.completions.create(
        messages=message_history, stream=True, **settings
    )

    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await msg.stream_token(token)

    message_history.append({"role": "assistant", "content": msg.content})
    await msg.update()

    summary = msg.content

    print("------------ DEBUGGING ------------")
    print("START SUMMARY RESULT")
    print(summary)
    print("END SUMMARY RESULT")
    print("------------ DEBUGGING ------------")    

    # Step 6: Check back with the user if they want to edit their summary before continuing
    summary_edit_check = await cl.AskActionMessage(
        content="Ist die Zusammenfassung so korrekt oder möchten Sie noch eine Änderung vornehmen?",
        actions=[
            cl.Action(name="Ja", value="Ja", label="Die Zusammenfassung ist korrekt."),
            cl.Action(name="Nein", value="Nein", label="Ich möchte noch etwas anpassen."),
        ],
        timeout=999999
    ).send()
    cl.user_session.set("summary_edit_check", summary_edit_check["value"])

    if (cl.user_session.get("summary_edit_check") == "Nein"):
        changes_to_summary = await cl.AskUserMessage(content="Welche Änderungen sollen vorgenommen werden?", timeout=999999).send()
        cl.user_session.set("changes_to_summary", changes_to_summary)   

        # Now the LLM needs to re-do the summary
        prompt_add_summary_input  = f"""
            {base_system_prompt}
            \n
            You have created the following summary off of the given information: {summary} \n
            The user now has the following adjustment remarks: \n
            {cl.user_session.get('changes_to_summary')['output']} \n
            \n
            Your task is to adjust the summary with those information accordingly. \n
            Your answers have to be at most 500 characters long. \n
            Never make any assumptions or add any information that is not explicitly stated by the user. \n 
        """
        message_history.append({"role": "user", "content": prompt_add_summary_input})    
        msg = cl.Message(content="")
        await msg.send()

        stream = await client.chat.completions.create(
            messages=message_history, stream=True, **settings
        )

        async for part in stream:
            if token := part.choices[0].delta.content or "":
                await msg.stream_token(token)

        message_history.append({"role": "assistant", "content": msg.content})
        await msg.update()

        summary = msg.content

    # Step 7: Translate the summary into english to get better results when performing the RAG-search
    prompt_translate_summary_to_english = f"""
        {base_system_prompt}
        \n
        You have created the following summary off of the given information: {summary} \n
        \n
        Your task is to translate this summary into the english language. \n
        Your answers have to be at most 500 characters long. \n
        Never make any assumptions or add any information that is not explicitly stated by the user. \n
    """
    message_history.append({"role": "user", "content": prompt_translate_summary_to_english})    

    stream = await client.chat.completions.create(
        messages=message_history, stream=True, **settings
    )

    message_history.append({"role": "assistant", "content": msg.content})
    await msg.update()

    summary = msg.content   
    print("------------ DEBUGGING ------------")
    print("START NEW SUMMARY RESULT")
    print(summary)
    print("END NEW SUMMARY RESULT")
    print("------------ DEBUGGING ------------")

    # Step 9: Evalute the idea based on RAG 2 (AI-Act and Whitepaper) (k=5 for now)
    await cl.Message(
        author="LexyAI", content="Ihre Idee wird nun durch das RAG-System bewertet. Hierfür wird der AI-Act sowie einschlägige Whitepaper gescannt und anschließend formuliert ein LLM eine Einschätzung."
    ).send()   

    query_engine = cl.user_session.get("query_engine") # type: RetrieverQueryEngine

    msg = cl.Message(content="", author="LexyAI")

    prompt_evaluate_idea = f"""
        {base_system_prompt}
        \n
        You have created the following summary off of the given information: {summary} \n
        \n
        Your task is to evaluate this summary with regard to the AI Act. \n
        First check whether the intended use falls under the AI Act. \n
        Then return recommendations for action and parameters that need to be taken into account. \n
        Always refer to the contextual information provided and the results of the search.
        Never make any assumptions or add any information that is not explicitly stated by the user. \n
    """

    res = await cl.make_async(query_engine.query)(prompt_evaluate_idea)

    for token in res.response_gen:
        await msg.stream_token(token)
    await msg.send()

@cl.on_message
async def main(message: cl.Message):
    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": message.content})

    msg = cl.Message(content="")
    await msg.send()

    stream = await client.chat.completions.create(
        messages=message_history, stream=True, **settings
    )

    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await msg.stream_token(token)

    print(msg.content)
    message_history.append({"role": "assistant", "content": msg.content})
    await msg.update()

