import { Injectable } from '@angular/core';
import { environment } from '../../environments/environment';
import {ApiAiClient} from 'api-ai-javascript';
import { Observable, BehaviorSubject } from 'rxjs';

 

export class Message{
  constructor(public content: string, public sentBy: string) {}


}

@Injectable({
  providedIn: 'root'
})
export class ChatService {

  readonly token = environment.dialogflow.instituteBot
  readonly client = new ApiAiClient({accessToken:this.token})

  constructor() { }

  conversation = new BehaviorSubject<Message[]>([]);

  update(msg: Message) {
    this.conversation.next([msg]);
  }

  converse(msg: string) {
    const UserMessage = new Message(msg, 'user');
    this.update(UserMessage);

    return this.client.textRequest(msg).then(res => {
      const speech =res.result.fulfillment.speech;
      const botMessage = new Message(speech ,'bot');
      this.update(botMessage)
    })
  }

  talk() {
    this.client.textRequest("who are you?").then(res => console.log(res))

  }
}
