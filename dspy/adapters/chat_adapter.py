import re
import textwrap
from typing import Any, NamedTuple

from litellm import ContextWindowExceededError
from pydantic.fields import FieldInfo

from dspy.adapters.base import Adapter
from dspy.adapters.utils import (
    format_field_value,
    get_annotation_name,
    get_field_description_string,
    parse_value,
    translate_field_type,
)
from dspy.clients.lm import LM
from dspy.signatures.signature import Signature
from dspy.utils.exceptions import AdapterParseError

field_header_pattern = re.compile(r"\[\[ ## (\w+) ## \]\]")


class FieldInfoWithName(NamedTuple):
    """Pairs a field name with its Pydantic FieldInfo metadata."""
    name: str
    info: FieldInfo


class ChatAdapter(Adapter):
    """Adapter that formats DSPy signatures as chat-based prompts with field markers.
    
    This adapter structures prompts using special markers `[[ ## field_name ## ]]` to delimit
    input and output fields. It supports fallback to JSONAdapter when parsing fails or when
    the language model doesn't follow the expected format.
    """
    def __init__(
        self,
        callbacks=None,
        use_native_function_calling: bool = False,
        native_response_types=None,
        use_json_adapter_fallback: bool = True,
    ):
        """Initialize the ChatAdapter with optional JSONAdapter fallback.
        
        Args:
            callbacks: List of callback functions for monitoring and logging.
            use_native_function_calling: Whether to enable native function calling.
            native_response_types: List of types to handle via native LM features.
            use_json_adapter_fallback: Whether to automatically fall back to JSONAdapter on errors.
        """
        super().__init__(
            callbacks=callbacks,
            use_native_function_calling=use_native_function_calling,
            native_response_types=native_response_types,
        )
        self.use_json_adapter_fallback = use_json_adapter_fallback

    def __call__(
        self,
        lm: LM,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Execute the adapter pipeline with automatic fallback to JSONAdapter on errors.
        
        Attempts to format and parse using ChatAdapter's field marker format. If any exception
        occurs (except ContextWindowExceededError), falls back to JSONAdapter unless disabled
        or already using JSONAdapter.
        
        Returns:
            List of dictionaries containing parsed output fields from the language model response.
            
        Raises:
            ContextWindowExceededError: When the input exceeds the model's context window.
            Exception: Any exception from the parent class if fallback is disabled or unavailable.
        """
        try:
            return super().__call__(lm, lm_kwargs, signature, demos, inputs)
        except Exception as e:
            # fallback to JSONAdapter
            from dspy.adapters.json_adapter import JSONAdapter

            if (
                isinstance(e, ContextWindowExceededError)
                or isinstance(self, JSONAdapter)
                or not self.use_json_adapter_fallback
            ):
                # On context window exceeded error, already using JSONAdapter, or use_json_adapter_fallback is False
                # we don't want to retry with a different adapter. Raise the original error instead of the fallback error.
                raise e
            return JSONAdapter()(lm, lm_kwargs, signature, demos, inputs)

    async def acall(
        self,
        lm: LM,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Async version of __call__ with the same fallback behavior.
        
        Returns:
            List of dictionaries containing parsed output fields from the language model response.
            
        Raises:
            ContextWindowExceededError: When the input exceeds the model's context window.
            Exception: Any exception from the parent class if fallback is disabled or unavailable.
        """
        try:
            return await super().acall(lm, lm_kwargs, signature, demos, inputs)
        except Exception as e:
            # fallback to JSONAdapter
            from dspy.adapters.json_adapter import JSONAdapter

            if (
                isinstance(e, ContextWindowExceededError)
                or isinstance(self, JSONAdapter)
                or not self.use_json_adapter_fallback
            ):
                # On context window exceeded error, already using JSONAdapter, or use_json_adapter_fallback is False
                # we don't want to retry with a different adapter. Raise the original error instead of the fallback error.
                raise e
            return await JSONAdapter().acall(lm, lm_kwargs, signature, demos, inputs)

    def format_field_description(self, signature: type[Signature]) -> str:
        """Format the field descriptions as separate input and output sections.
        
        Returns:
            A string containing formatted descriptions of all input and output fields.
        """
        return (
            f"Your input fields are:\n{get_field_description_string(signature.input_fields)}\n"
            f"Your output fields are:\n{get_field_description_string(signature.output_fields)}"
        )

    def format_field_structure(self, signature: type[Signature]) -> str:
        """Format the expected structure using field markers like [[ ## field_name ## ]].
        
        Creates a template showing how input and output fields should be formatted, with each
        field wrapped in markers. Adds a `[[ ## completed ## ]]` marker at the end to signal
        completion of the output section.
        
        Returns:
            A string template showing the expected structure with field markers.
        """
        parts = []
        parts.append("All interactions will be structured in the following way, with the appropriate values filled in.")

        def format_signature_fields_for_instructions(fields: dict[str, FieldInfo]):
            """Convert signature fields into formatted field markers with type information."""
            return self.format_field_with_value(
                fields_with_values={
                    FieldInfoWithName(name=field_name, info=field_info): translate_field_type(field_name, field_info)
                    for field_name, field_info in fields.items()
                },
            )

        parts.append(format_signature_fields_for_instructions(signature.input_fields))
        parts.append(format_signature_fields_for_instructions(signature.output_fields))
        parts.append("[[ ## completed ## ]]\n")
        return "\n\n".join(parts).strip()

    def format_task_description(self, signature: type[Signature]) -> str:
        """Format the task description from the signature's instructions.
        
        Returns:
            A string containing the formatted task description with proper indentation.
        """
        instructions = textwrap.dedent(signature.instructions)
        objective = ("\n" + " " * 8).join([""] + instructions.splitlines())
        return f"In adhering to this structure, your objective is: {objective}"

    def format_user_message_content(
        self,
        signature: type[Signature],
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> str:
        """Format user message content with input field values wrapped in markers.
        
        Constructs a message containing all input fields formatted as `[[ ## field_name ## ]]`
        followed by the field value. Optionally includes output format requirements when this
        is the main request.
        
        Returns:
            A formatted string containing all input fields with their values and optional format requirements.
        """
        messages = [prefix]
        for k, v in signature.input_fields.items():
            if k in inputs:
                value = inputs.get(k)
                formatted_field_value = format_field_value(field_info=v, value=value)
                messages.append(f"[[ ## {k} ## ]]\n{formatted_field_value}")

        if main_request:
            output_requirements = self.user_message_output_requirements(signature)
            if output_requirements is not None:
                messages.append(output_requirements)

        messages.append(suffix)
        return "\n\n".join(messages).strip()

    def user_message_output_requirements(self, signature: type[Signature]) -> str:
        """Returns a simplified format reminder for the language model.

        In chat-based interactions, language models may lose track of the required output format
        as the conversation context grows longer. This method generates a concise reminder of
        the expected output structure that can be included in user messages.
        
        Returns:
            A string describing the expected output format with field markers and type requirements.
        """

        def type_info(v):
            """Generate type requirement text for non-string fields."""
            if v.annotation is not str:
                return f" (must be formatted as a valid Python {get_annotation_name(v.annotation)})"
            else:
                return ""

        message = "Respond with the corresponding output fields, starting with the field "
        message += ", then ".join(f"`[[ ## {f} ## ]]`{type_info(v)}" for f, v in signature.output_fields.items())
        message += ", and then ending with the marker for `[[ ## completed ## ]]`."
        return message

    def format_assistant_message_content(
        self,
        signature: type[Signature],
        outputs: dict[str, Any],
        missing_field_message=None,
    ) -> str:
        """Format assistant message content with output field values wrapped in markers.
        
        Constructs a message containing all output fields formatted as `[[ ## field_name ## ]]`
        followed by the field value. Ends with the completion marker `[[ ## completed ## ]]`.
        
        Args:
            missing_field_message: Optional message to use when an output field value is missing.
            
        Returns:
            A formatted string containing all output fields with their values and the completion marker.
        """
        assistant_message_content = self.format_field_with_value(
            {
                FieldInfoWithName(name=k, info=v): outputs.get(k, missing_field_message)
                for k, v in signature.output_fields.items()
            },
        )
        assistant_message_content += "\n\n[[ ## completed ## ]]\n"
        return assistant_message_content

    def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        """Parse the LM completion by extracting content between field markers.
        
        Scans the completion for field markers `[[ ## field_name ## ]]` and extracts the content
        following each marker until the next marker. Validates that all expected output fields
        are present and parses each field value according to its type annotation.
        
        Returns:
            A dictionary mapping output field names to their parsed values.
            
        Raises:
            AdapterParseError: If parsing fails for any field or if expected fields are missing.
        """
        sections = [(None, [])]

        for line in completion.splitlines():
            match = field_header_pattern.match(line.strip())
            if match:
                # If the header pattern is found, split the rest of the line as content
                header = match.group(1)
                remaining_content = line[match.end() :].strip()
                sections.append((header, [remaining_content] if remaining_content else []))
            else:
                sections[-1][1].append(line)

        sections = [(k, "\n".join(v).strip()) for k, v in sections]

        fields = {}
        for k, v in sections:
            if (k not in fields) and (k in signature.output_fields):
                try:
                    fields[k] = parse_value(v, signature.output_fields[k].annotation)
                except Exception as e:
                    raise AdapterParseError(
                        adapter_name="ChatAdapter",
                        signature=signature,
                        lm_response=completion,
                        message=f"Failed to parse field {k} with value {v} from the LM response. Error message: {e}",
                    )
        if fields.keys() != signature.output_fields.keys():
            raise AdapterParseError(
                adapter_name="ChatAdapter",
                signature=signature,
                lm_response=completion,
                parsed_result=fields,
            )

        return fields

    def format_field_with_value(self, fields_with_values: dict[FieldInfoWithName, Any]) -> str:
        """Format multiple fields with their values, each wrapped in field markers.
        
        Converts a dictionary of fields and values into a multi-line string where each field
        is formatted as `[[ ## field_name ## ]]` followed by the formatted field value.
        
        Returns:
            A formatted string with all fields and their values separated by field markers.
        """
        output = []
        for field, field_value in fields_with_values.items():
            formatted_field_value = format_field_value(field_info=field.info, value=field_value)
            output.append(f"[[ ## {field.name} ## ]]\n{formatted_field_value}")

        return "\n\n".join(output).strip()

    def format_finetune_data(
        self,
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
        outputs: dict[str, Any],
    ) -> dict[str, list[Any]]:
        """Format the call data into OpenAI-compatible fine-tuning format.
        
        Converts the signature, demos, inputs, and outputs into a list of chat messages with
        roles (system, user, assistant). The system and user messages are generated from the
        format method, and the assistant message is generated from the outputs.
        
        Returns:
            A dictionary with a "messages" key containing the formatted chat messages.
        """
        system_user_messages = self.format(  # returns a list of dicts with the keys "role" and "content"
            signature=signature, demos=demos, inputs=inputs
        )
        assistant_message_content = self.format_assistant_message_content(  # returns a string, without the role
            signature=signature, outputs=outputs
        )
        assistant_message = {"role": "assistant", "content": assistant_message_content}
        messages = system_user_messages + [assistant_message]
        return {"messages": messages}
