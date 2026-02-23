import gradio as gr
from src.ui.interface import create_ui

if __name__ == "__main__":
    demo, custom_css = create_ui()
    demo.launch(
        theme=gr.themes.Default(primary_hue="blue", spacing_size="sm"),
        css=custom_css,
        server_name="0.0.0.0"
    )
