from Step3_rag_traffic_law_bot import create_rag_graph

def main():
    graph = create_rag_graph()          # tạo graph chưa compile
    mermaid = graph.get_graph().draw_mermaid()  # xuất Mermaid
    with open("rag_graph.mmd", "w", encoding="utf-8") as f:
        f.write(mermaid)
    print("Exported rag_graph.mmd")

if __name__ == "__main__":
    main()