from autogen import ConversableAgent
import os


def save_reply_decorator(func):
    """
    Decorator to save replies to a file.
    This will wrap around generate_reply to save replies after they're generated.
    """

    def wrapper(self, *args, **kwargs):
        # Call the original function
        reply = func(self, *args, **kwargs)

        # Now, save the reply using the logic you provided
        if self.reply_save_path and reply:
            self.save_reply_custom(reply)

        return reply

    return wrapper


class CustomConversableAgent(ConversableAgent):
    def __init__(self, *args, reply_save_path="", **kwargs):
        super().__init__(*args, **kwargs)
        self.reply_save_path = reply_save_path

    def save_reply_custom(self, reply):
        """Custom method to save reply to a file."""
        directory = os.path.dirname(self.reply_save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        content = None
        if "```python" in reply:
            start = reply.find("```python") + len("```python")
            end = reply.find("```", start)
            if end != -1:
                content = reply[start:end].strip()

            if content:
                with open(self.reply_save_path, 'w') as file:
                    file.write(content + '\n')

    @save_reply_decorator
    def generate_reply(self, *args, **kwargs):
        return super().generate_reply(*args, **kwargs)
