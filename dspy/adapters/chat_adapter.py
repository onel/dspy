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
from dspy.utils.callback import BaseCallback
from dspy.utils.exceptions import AdapterParseError

field_header_pattern = re.compile(r"\[\[ ## (\w+) ## \]\]")


class FieldInfoWithName(NamedTuple):
    """Pairs a field name with its Pydantic FieldInfo metadata.
    
    This structure is used internally to track field information along with
    the field's name when formatting output.
    """
    name: str
    info: FieldInfo


class ChatAdapter(Adapter):
    """Default Adapter for most language models.

    The ChatAdapter formats DSPy signatures into a format compatible with most language models.
    It uses delimiter patterns like `[[ ## field_name ## ]]` to clearly separate input and output fields in
    the message content.

    Key features:
        - Structures inputs and outputs using field header markers for clear field delineation.
        - Provides automatic fallback to JSONAdapter if the chat format fails.
    """

    def __init__(
        self,
        callbacks: list[BaseCallback] | None = None,
        use_native_function_calling: bool = False,
        native_response_types: list[type[type]] | None = None,
        use_json_adapter_fallback: bool = True,
    ):
        """
        Args:
            callbacks: List of callback functions to execute during adapter methods.
            use_native_function_calling: Whether to enable native function calling capabilities.
            native_response_types: List of output field types handled by native LM features.
            use_json_adapter_fallback: Whether to automatically fallback to JSONAdapter if the ChatAdapter fails.
                If True, when an error occurs (except ContextWindowExceededError), the adapter will retry using
                JSONAdapter. Defaults to True.
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
        """Execute the adapter pipeline with automatic fallback to JSONAdapter on failure.
        
        Attempts to format inputs, call the LM, and parse outputs using the chat format.
        If an error occurs (except ContextWindowExceededError), automatically retries
        using JSONAdapter when fallback is enabled.
        
        Returns:
            List of dictionaries containing the parsed output fields from the LM response.
            
        Raises:
            ContextWindowExceededError: When the input exceeds the model's context window.
            Exception: Any exception from the parent class call if fallback is disabled or
                if already using JSONAdapter.
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
        """Async version of the adapter pipeline with automatic fallback to JSONAdapter on failure.
        
        Attempts to format inputs, call the LM asynchronously, and parse outputs using the chat format.
        If an error occurs (except ContextWindowExceededError), automatically retries
        using JSONAdapter when fallback is enabled.
        
        Returns:
            List of dictionaries containing the parsed output fields from the LM response.
            
        Raises:
            ContextWindowExceededError: When the input exceeds the model's context window.
            Exception: Any exception from the parent class call if fallback is disabled or
                if already using JSONAdapter.
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
        """Format field descriptions as numbered lists for input and output fields.
        
        Creates a two-section description listing all input fields followed by all output fields,
        with each field numbered and including its type annotation and description.
        
        Returns:
            A formatted string with two sections: input fields and output fields, each with
            numbered entries containing field names, types, and descriptions.
        """
        return (
            f"Your input fields are:\n{get_field_description_string(signature.input_fields)}\n"
            f"Your output fields are:\n{get_field_description_string(signature.output_fields)}"
        )

    def format_field_structure(self, signature: type[Signature]) -> str:
        """Format the expected structure for input and output fields using field header markers.
        
        Creates a template showing how fields should be structured in the conversation,
        with each field preceded by a `[[ ## field_name ## ]]` marker. Adds a `completed`
        marker at the end to signal the end of output fields.
        
        Returns:
            A formatted string showing the expected structure with field header markers for
            all input and output fields, ending with a completion marker.
        """
        parts = []
        parts.append("All interactions will be structured in the following way, with the appropriate values filled in.")

        def format_signature_fields_for_instructions(fields: dict[str, FieldInfo]):
            """Format signature fields into a structured template with field headers.
            
            Converts each field into a formatted string with field header markers and
            translated type information suitable for instruction display.
            
            Returns:
                A formatted string with field headers and type information for each field.
            """
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
        
        Wraps the signature instructions with an introductory phrase and proper indentation.
        
        Returns:
            A formatted string containing the task objective with proper indentation.
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
        """Format user message content with field headers and optional output requirements.
        
        Constructs the user message by adding field header markers before each input field value.
        When this is the main request, appends a reminder of the expected output format.
        
        Returns:
            A formatted string containing the user message with field headers, optional prefix/suffix,
            and output requirements if this is the main request.
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

        Args:
            signature (Type[Signature]): The DSPy signature defining the expected input/output fields.

        Returns:
            str: A simplified description of the required output format.

        Note:
            This is a more lightweight version of `format_field_structure` specifically designed
            for inline reminders within chat messages.
        """

        def type_info(v):
            """Generate type information string for a field.
            
            Returns an empty string for string types, otherwise returns a note about
            the required Python type format.
            
            Returns:
                A string describing the required format, or empty string for string types.
            """
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
        """Format assistant message content with field headers and a completion marker.
        
        Constructs the assistant message by adding field header markers before each output field value.
        Appends a `completed` marker at the end to signal the end of the response.
        
        Args:
            missing_field_message: The message to use for fields that are missing from outputs.
                If None, the field will be included without a value.
        
        Returns:
            A formatted string containing the assistant message with field headers and a
            completion marker at the end.
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
        """Parse LM completion by extracting values between field header markers.
        
        Splits the completion text into sections based on field header patterns,
        then parses each section's content according to the field's type annotation.
        Validates that all expected output fields are present.
        
        Returns:
            A dictionary mapping field names to their parsed values from the completion.
            
        Raises:
            AdapterParseError: If a field value cannot be parsed according to its type
                annotation, or if the parsed fields don't match the expected output fields.
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
        """
        Formats the values of the specified fields according to the field's DSPy type (input or output),
        annotation (e.g. str, int, etc.), and the type of the value itself. Joins the formatted values
        into a single string, which is is a multiline string if there are multiple fields.

        Args:
            fields_with_values: A dictionary mapping information about a field to its corresponding
                value.

        Returns:
            The joined formatted values of the fields, represented as a string
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
        """Format call data into OpenAI-compatible finetuning format.
        
        Converts the signature, demos, inputs, and outputs into a list of messages
        with system, user, and assistant roles. Returns the messages wrapped in
        a dictionary with a "messages" key.
        
        Returns:
            A dictionary with a "messages" key containing a list of message dictionaries,
            each with "role" and "content" keys formatted for OpenAI finetuning.
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
