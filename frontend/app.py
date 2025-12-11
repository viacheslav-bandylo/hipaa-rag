import os
import gradio as gr
import httpx

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


async def check_status():
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{BACKEND_URL}/api/ingest/status", timeout=10.0)
            data = response.json()
            if data["status"] == "ready":
                return f"Database ready with {data['indexed_chunks']} indexed chunks"
            else:
                return "Documents not indexed. Click 'Ingest Documents' to start."
        except Exception as e:
            return f"Error checking status: {str(e)}"


async def ingest_documents():
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{BACKEND_URL}/api/ingest",
                timeout=300.0
            )
            if response.status_code == 200:
                data = response.json()
                return f"Success! {data['message']}"
            else:
                return f"Error: {response.json().get('detail', 'Unknown error')}"
        except Exception as e:
            return f"Error during ingestion: {str(e)}"


async def chat(message, history):
    if not message.strip():
        return history, ""

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{BACKEND_URL}/api/chat",
                json={"message": message},
                timeout=60.0
            )

            if response.status_code == 200:
                data = response.json()
                answer = data["answer"]

                if data.get("sources"):
                    answer += "\n\n---\n**Sources:**\n"
                    for i, source in enumerate(data["sources"], 1):
                        answer += f"\n{i}. **{source['section_reference']}** (Page {source.get('page_number', 'N/A')})"

                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": answer})
            else:
                error_detail = response.json().get("detail", "Unknown error")
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": f"Error: {error_detail}"})

        except httpx.TimeoutException:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": "Request timed out. Please try again."})
        except Exception as e:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": f"Error: {str(e)}"})

    return history, ""


def clear_chat():
    return [], ""


with gr.Blocks(
    title="HIPAA RAG Assistant"
) as demo:
    gr.Markdown(
        """
        # HIPAA Compliance Assistant

        Ask questions about HIPAA regulations (Parts 160, 162, and 164).
        The assistant will provide answers with citations to specific sections.

        **Example questions:**
        - What are the requirements for a covered entity under HIPAA?
        - What is the definition of protected health information (PHI)?
        - What are the penalties for HIPAA violations?
        - Quote the exact text of §164.502(a)
        """
    )

    with gr.Row():
        with gr.Column(scale=4):
            status_text = gr.Textbox(
                label="System Status",
                interactive=False,
                value="Checking status..."
            )
        with gr.Column(scale=1):
            ingest_btn = gr.Button("Ingest Documents", variant="secondary")
            refresh_btn = gr.Button("Refresh Status", variant="secondary")

    chatbot = gr.Chatbot(
        label="Chat",
        height=500
    )

    with gr.Row():
        msg_input = gr.Textbox(
            label="Your Question",
            placeholder="Ask a question about HIPAA regulations...",
            scale=4,
            show_label=False
        )
        submit_btn = gr.Button("Send", variant="primary", scale=1)

    clear_btn = gr.Button("Clear Chat", variant="secondary")

    submit_btn.click(
        chat,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, msg_input]
    )

    msg_input.submit(
        chat,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, msg_input]
    )

    clear_btn.click(
        clear_chat,
        outputs=[chatbot, msg_input]
    )

    ingest_btn.click(
        ingest_documents,
        outputs=[status_text]
    )

    refresh_btn.click(
        check_status,
        outputs=[status_text]
    )

    demo.load(
        check_status,
        outputs=[status_text]
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft()
    )
