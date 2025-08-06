# hpii-chatbot
# Database chat history
Create DB
CREATE TABLE chat_history (
	id serial4 NOT NULL,
	user_id text NOT NULL,
	question text NOT NULL,
	answer text NOT NULL,
	created_at timestamp DEFAULT now() NULL,
	CONSTRAINT chat_history_pkey PRIMARY KEY (id)
);

# Setup 
# Config.ini :Setup koneksi postgress
