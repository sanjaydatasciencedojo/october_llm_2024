import utils
import streamlit as st
from langchain.agents import AgentType
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import AgentExecutor, create_tool_calling_agent, Tool, initialize_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain.prompts import ChatPromptTemplate



st.set_page_config(page_title="ChatWeb", page_icon="🌐")
st.header('Chatbot with Web Browser Access')
st.write('Equipped with internet agent, enables users to ask questions about recent events')


class ChatbotTools:

    def __init__(self):
        utils.configure_openai_api_key()
        self.openai_model = "gpt-4o-mini"

    def setup_agent(self):
        # Define tool
        ddg_search = DuckDuckGoSearchRun()
        tools = [
            Tool(
                name="DuckDuckGoSearch",
                func=ddg_search.run,
                description="Useful for when you need to answer questions about current events. You should ask targeted questions",
            )
        ]

        # Setup LLM and Agent
        llm = ChatOpenAI(model_name=self.openai_model, streaming=True)
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            verbose=True
        )

        return agent

    @utils.enable_chat_history
    def main(self):
        agent = self.setup_agent()
        user_query = st.chat_input(placeholder="Ask me anything!")
        if user_query:
            utils.display_msg(user_query, 'user')
            with st.chat_message("assistant"):
                try:
                    st_cb = StreamlitCallbackHandler(st.container())
                    response = agent.run(user_query, callbacks=[st_cb])
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response})
                    st.write(response)
                except Exception as e:
                    print(e)



if __name__ == "__main__":
    obj = ChatbotTools()
    obj.main()
